from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h2r_det import H2RConfig, H2RDetector, H2RLoss, compute_map50, decode_predictions, mean_routed_area_fraction, routing_recall
from h2r_det.utils import (
    ModelEMA,
    checkpoint_payload,
    cleanup_distributed,
    ensure_dir,
    human_only_targets,
    init_distributed,
    is_distributed,
    is_main_process,
    move_targets_to_device,
    promote_fp32_tree,
    set_seed,
    write_json,
)
from h2r_det.visdrone import build_visdrone_dataloader, load_visdrone_yaml


def parse_args() -> argparse.Namespace:
    defaults = H2RConfig()
    parser = argparse.ArgumentParser(description="Train H2R-Det on a VisDrone YAML split.")
    parser.add_argument("--visdrone-yaml", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-process batch size. Global batch = batch_size * world_size.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--patch-size", type=int, default=defaults.patch_size)
    parser.add_argument("--max-routes", type=int, default=defaults.max_routes)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--scout-score-thresh", type=float, default=0.05)
    parser.add_argument("--refine-score-thresh", type=float, default=0.1)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable AMP on CUDA.")
    parser.add_argument(
        "--find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DDP find_unused_parameters. Keep disabled unless you hit a DDP hang from dynamic branches.",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many validation epochs without improvement. 0 disables early stopping.",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-metric", type=str, default="human_ap50", choices=("human_ap50", "map50", "loss"))
    parser.add_argument("--min-epochs", type=int, default=0, help="Do not early-stop before this epoch.")
    return parser.parse_args()


def _cpu_batch(items: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{key: value.detach().cpu() for key, value in item.items()} for item in items]


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _dist_barrier(local_rank: int, device: torch.device) -> None:
    if not dist.is_initialized():
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _reduce_mean(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _build_scheduler(optimizer: AdamW, epochs: int, config: H2RConfig) -> LambdaLR:
    warmup_epochs = max(0, config.warmup_epochs)
    min_lr_ratio = float(config.min_lr_ratio)

    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return max(1e-3, float(epoch_idx + 1) / float(warmup_epochs))
        if epochs <= warmup_epochs:
            return 1.0
        progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _metric_improved(metric_name: str, current: float, best: float, min_delta: float) -> bool:
    if math.isinf(best):
        return True
    if metric_name == "loss":
        return current < (best - min_delta)
    return current > (best + min_delta)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: H2RLoss,
    loader,
    sampler,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: H2RConfig,
    use_amp: bool,
    epoch: int,
    log_interval: int,
    ema: ModelEMA | None,
) -> dict[str, float]:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss = 0.0
    total_route_recall = 0.0
    total_area = 0.0
    total_routes_per_image = 0.0
    batches = 0
    skipped_batches = 0

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, teacher_targets=human_only_targets(config, targets))
        losses = criterion(
            promote_fp32_tree(outputs),
            targets,
            image_size=(config.image_size, config.image_size),
        )

        if not torch.isfinite(losses["total"]):
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            skipped_batches += 1
            continue
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(_unwrap_model(model))

        total_loss += float(losses["total"].item())
        total_route_recall += routing_recall(config, outputs["routes"], targets)
        total_area += mean_routed_area_fraction(outputs["routes"], (config.image_size, config.image_size))
        total_routes_per_image += outputs["routes"].predicted_count / max(1, images.shape[0])
        batches += 1

        if is_main_process() and (batch_idx == 1 or batch_idx % log_interval == 0):
            print(
                json.dumps(
                    {
                        "event": "train_batch",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss": float(losses["total"].item()),
                        "grad_norm": float(torch.as_tensor(grad_norm).item()),
                        "routes_per_image": float(outputs["routes"].predicted_count / max(1, images.shape[0])),
                        "skipped_batches": skipped_batches,
                    }
                )
            )

    stats = {
        "loss": total_loss / max(1, batches),
        "route_recall": total_route_recall / max(1, batches),
        "routed_area": total_area / max(1, batches),
        "routes_per_image": total_routes_per_image / max(1, batches),
        "skipped_batches": float(skipped_batches),
    }
    return {key: _reduce_mean(value, device) for key, value in stats.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: H2RLoss,
    loader,
    device: torch.device,
    config: H2RConfig,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_route_recall = 0.0
    total_area = 0.0
    total_routes_per_image = 0.0
    batches = 0
    predictions_all: list[dict[str, torch.Tensor]] = []
    targets_all: list[dict[str, torch.Tensor]] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
        losses = criterion(
            promote_fp32_tree(outputs),
            targets,
            image_size=(config.image_size, config.image_size),
        )

        predictions = decode_predictions(
            promote_fp32_tree(outputs),
            config,
            image_size=(config.image_size, config.image_size),
            scout_score_thresh=args.scout_score_thresh,
            refine_score_thresh=args.refine_score_thresh,
            topk=args.topk,
            nms_iou=args.nms_iou,
        )
        predictions_all.extend(_cpu_batch(predictions))
        targets_all.extend(_cpu_batch(targets))
        total_loss += float(losses["total"].item())
        total_route_recall += routing_recall(config, outputs["routes"], targets)
        total_area += mean_routed_area_fraction(outputs["routes"], (config.image_size, config.image_size))
        total_routes_per_image += outputs["routes"].predicted_count / max(1, images.shape[0])
        batches += 1

    metrics = compute_map50(predictions_all, targets_all, config.num_classes)
    human_ap = sum(metrics["ap50_per_class"][idx] for idx in config.human_class_ids) / len(config.human_class_ids)
    return {
        "loss": total_loss / max(1, batches),
        "route_recall": total_route_recall / max(1, batches),
        "routed_area": total_area / max(1, batches),
        "routes_per_image": total_routes_per_image / max(1, batches),
        "map50": float(metrics["map50"]),
        "human_ap50": float(human_ap),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device, rank, world_size, local_rank = init_distributed(args.device)
    try:
        dataset_root = args.dataset_root or None
        use_amp = bool(args.amp and device.type == "cuda")

        if is_distributed():
            parsed = load_visdrone_yaml(args.visdrone_yaml, override_root=dataset_root) if is_main_process() else None
            dist.barrier()
            if not is_main_process():
                parsed = load_visdrone_yaml(args.visdrone_yaml, override_root=dataset_root)
            assert parsed is not None
        else:
            parsed = load_visdrone_yaml(args.visdrone_yaml, override_root=dataset_root)
        run_name = args.run_name or f"h2r_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = ensure_dir(Path(args.output_dir) / run_name)

        config = H2RConfig(
            num_classes=parsed.nc,
            class_names=parsed.names,
            image_size=args.image_size,
            patch_size=args.patch_size,
            max_routes=args.max_routes,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        train_loader, train_sampler = build_visdrone_dataloader(
            args.visdrone_yaml,
            split=args.train_split,
            image_size=config.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            limit=args.limit_train or None,
            override_root=dataset_root,
            distributed=is_distributed(),
            drop_last=is_distributed(),
        )
        val_loader = None
        if is_main_process():
            val_loader, _ = build_visdrone_dataloader(
                args.visdrone_yaml,
                split=args.val_split,
                image_size=config.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                limit=args.limit_val or None,
                override_root=dataset_root,
                distributed=False,
                drop_last=False,
            )

        base_model = H2RDetector(config).to(device)
        model: torch.nn.Module = base_model
        if is_distributed():
            model = DDP(
                base_model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=args.find_unused_parameters,
            )

        criterion = H2RLoss(config)
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        scheduler = _build_scheduler(optimizer, args.epochs, config)
        ema = ModelEMA(_unwrap_model(model), decay=config.ema_decay) if is_main_process() else None

        if is_main_process():
            write_json(
                run_dir / "config.json",
                {
                    "args": vars(args),
                    "model_config": asdict(config),
                    "distributed": {"world_size": world_size, "rank": rank, "local_rank": local_rank},
                    "dataset_summary": {
                        "root": str(parsed.root),
                        "train": str(getattr(parsed, args.train_split)),
                        "val": str(getattr(parsed, args.val_split)),
                    },
                },
            )
            print(
                json.dumps(
                    {
                        "device": str(device),
                        "world_size": world_size,
                        "per_gpu_batch_size": args.batch_size,
                        "global_batch_size": args.batch_size * world_size,
                        "amp": use_amp,
                        "base_lr": config.learning_rate,
                        "run_dir": str(run_dir),
                    }
                )
            )

        best_human_ap50 = float("-inf")
        early_stop_metric = args.early_stop_metric
        best_stop_metric = float("inf") if early_stop_metric == "loss" else float("-inf")
        epochs_without_improvement = 0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, args.epochs + 1):
            if is_main_process():
                print(json.dumps({"event": "epoch_start", "epoch": epoch, "epochs": args.epochs}))
            train_metrics = train_one_epoch(
                model,
                criterion,
                train_loader,
                train_sampler,
                optimizer,
                scaler,
                device,
                config,
                use_amp,
                epoch,
                args.log_interval,
                ema,
            )
            scheduler.step()
            _dist_barrier(local_rank, device)

            if is_main_process():
                assert val_loader is not None
                eval_model = ema.ema if ema is not None else _unwrap_model(model)
                val_metrics = evaluate(eval_model, criterion, val_loader, device, config, args)
                stop_metric_value = float(val_metrics[early_stop_metric])
                row = {
                    "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                history.append(row)
                print(json.dumps(row))

                model_to_save = ema.ema if ema is not None else _unwrap_model(model)
                torch.save(
                    checkpoint_payload(model_to_save, optimizer, config, epoch, val_metrics),
                    run_dir / "last.pt",
                )
                if val_metrics["human_ap50"] > best_human_ap50:
                    best_human_ap50 = val_metrics["human_ap50"]
                if _metric_improved(early_stop_metric, stop_metric_value, best_stop_metric, args.early_stop_min_delta):
                    best_stop_metric = stop_metric_value
                    epochs_without_improvement = 0
                    torch.save(
                        checkpoint_payload(model_to_save, optimizer, config, epoch, val_metrics),
                        run_dir / "best.pt",
                    )
                else:
                    epochs_without_improvement += 1

                write_json(
                    run_dir / "history.json",
                    {
                        "history": history,
                        "best_human_ap50": best_human_ap50,
                        "early_stop_metric": early_stop_metric,
                        "best_early_stop_metric": best_stop_metric,
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                )

                stop_training = (
                    args.early_stop_patience > 0
                    and epoch >= max(1, args.min_epochs)
                    and epochs_without_improvement >= args.early_stop_patience
                )
                if stop_training:
                    print(
                        json.dumps(
                            {
                                "event": "early_stop",
                                "epoch": epoch,
                                "metric": early_stop_metric,
                                "best_metric": best_stop_metric,
                                "epochs_without_improvement": epochs_without_improvement,
                                "patience": args.early_stop_patience,
                                "min_delta": args.early_stop_min_delta,
                            }
                        )
                    )
            else:
                stop_training = False

            if is_distributed():
                stop_tensor = torch.tensor([1 if stop_training else 0], device=device, dtype=torch.int32)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if stop_training:
                break

            _dist_barrier(local_rank, device)
    finally:
        cleanup_distributed()

    if is_main_process():
        print(f"Best human AP50: {best_human_ap50:.4f}")
        print(f"Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
