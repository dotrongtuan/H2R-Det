from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h2r_det import H2RConfig, H2RDetector, H2RLoss, compute_map50, decode_predictions, mean_routed_area_fraction, routing_recall
from h2r_det.utils import checkpoint_payload, ensure_dir, human_only_targets, move_targets_to_device, set_seed, write_json
from h2r_det.visdrone import build_visdrone_dataloader, load_visdrone_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train H2R-Det on a VisDrone YAML split.")
    parser.add_argument("--visdrone-yaml", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--max-routes", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--scout-score-thresh", type=float, default=0.25)
    parser.add_argument("--refine-score-thresh", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=150)
    return parser.parse_args()


def _cpu_batch(items: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{key: value.detach().cpu() for key, value in item.items()} for item in items]


def train_one_epoch(
    model: H2RDetector,
    criterion: H2RLoss,
    loader,
    optimizer: AdamW,
    device: torch.device,
    config: H2RConfig,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_route_recall = 0.0
    total_area = 0.0
    batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = move_targets_to_device(targets, device)
        outputs = model(images, teacher_targets=human_only_targets(config, targets))
        losses = criterion(outputs, targets, image_size=(config.image_size, config.image_size))

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()

        total_loss += float(losses["total"].item())
        total_route_recall += routing_recall(config, outputs["routes"], targets)
        total_area += mean_routed_area_fraction(outputs["routes"], (config.image_size, config.image_size))
        batches += 1

    return {
        "loss": total_loss / max(1, batches),
        "route_recall": total_route_recall / max(1, batches),
        "routed_area": total_area / max(1, batches),
    }


@torch.no_grad()
def evaluate(
    model: H2RDetector,
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
    batches = 0
    predictions_all: list[dict[str, torch.Tensor]] = []
    targets_all: list[dict[str, torch.Tensor]] = []

    for images, targets in loader:
        images = images.to(device)
        targets = move_targets_to_device(targets, device)
        outputs = model(images)
        losses = criterion(outputs, targets, image_size=(config.image_size, config.image_size))

        predictions = decode_predictions(
            outputs,
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
        batches += 1

    metrics = compute_map50(predictions_all, targets_all, config.num_classes)
    human_ap = sum(metrics["ap50_per_class"][idx] for idx in config.human_class_ids) / len(config.human_class_ids)
    return {
        "loss": total_loss / max(1, batches),
        "route_recall": total_route_recall / max(1, batches),
        "routed_area": total_area / max(1, batches),
        "map50": float(metrics["map50"]),
        "human_ap50": float(human_ap),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    dataset_root = args.dataset_root or None

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

    train_loader = build_visdrone_dataloader(
        args.visdrone_yaml,
        split=args.train_split,
        image_size=config.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        limit=args.limit_train or None,
        override_root=dataset_root,
    )
    val_loader = build_visdrone_dataloader(
        args.visdrone_yaml,
        split=args.val_split,
        image_size=config.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        limit=args.limit_val or None,
        override_root=dataset_root,
    )

    model = H2RDetector(config).to(device)
    criterion = H2RLoss(config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    write_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "model_config": asdict(config),
            "dataset_summary": {
                "root": str(parsed.root),
                "train": str(getattr(parsed, args.train_split)),
                "val": str(getattr(parsed, args.val_split)),
            },
        },
    )

    best_human_ap50 = float("-inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, device, config)
        val_metrics = evaluate(model, criterion, val_loader, device, config, args)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row))

        torch.save(
            checkpoint_payload(model, optimizer, config, epoch, val_metrics),
            run_dir / "last.pt",
        )
        if val_metrics["human_ap50"] > best_human_ap50:
            best_human_ap50 = val_metrics["human_ap50"]
            torch.save(
                checkpoint_payload(model, optimizer, config, epoch, val_metrics),
                run_dir / "best.pt",
            )

        write_json(run_dir / "history.json", {"history": history, "best_human_ap50": best_human_ap50})

    print(f"Best human AP50: {best_human_ap50:.4f}")
    print(f"Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
