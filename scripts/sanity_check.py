from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h2r_det import H2RConfig, H2RDetector, H2RLoss
from h2r_det.synthetic import generate_synthetic_batch
from h2r_det.visdrone import load_visdrone_yaml, summarize_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check the H2R-Det prototype.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--visdrone-yaml", type=str, default="")
    parser.add_argument("--max-routes", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--patch-size", type=int, default=80)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _human_targets(config: H2RConfig, targets: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    human_only = []
    for target in targets:
        boxes = target["boxes"]
        labels = target["labels"]
        if boxes.numel() == 0:
            human_only.append({"boxes": boxes, "labels": labels})
            continue
        human_mask = torch.isin(labels, labels.new_tensor(config.human_class_ids))
        human_only.append({"boxes": boxes[human_mask], "labels": labels[human_mask]})
    return human_only


def routing_recall(config: H2RConfig, routes, targets: list[dict[str, torch.Tensor]]) -> float:
    total = 0
    covered = 0
    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"]
        labels = target["labels"]
        if boxes.numel() == 0:
            continue
        human_mask = torch.isin(labels, labels.new_tensor(config.human_class_ids))
        boxes = boxes[human_mask]
        if boxes.numel() == 0:
            continue
        centers = torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0), dim=1)
        batch_routes = routes.rois[routes.rois[:, 0] == float(batch_idx)]
        total += centers.shape[0]
        if batch_routes.numel() == 0:
            continue
        for center in centers:
            inside = (
                (center[0] >= batch_routes[:, 1])
                & (center[0] <= batch_routes[:, 3])
                & (center[1] >= batch_routes[:, 2])
                & (center[1] <= batch_routes[:, 4])
            )
            covered += int(inside.any().item())
    return covered / max(1, total)


def area_fraction(routes, image_size: int) -> float:
    if routes.rois.numel() == 0:
        return 0.0
    areas = (routes.rois[:, 3] - routes.rois[:, 1]) * (routes.rois[:, 4] - routes.rois[:, 2])
    return float((areas / float(image_size * image_size)).mean().item())


def measure_patch_expert_flops(model: H2RDetector, patches: torch.Tensor) -> int:
    if patches.numel() == 0:
        return 0
    with FlopCounterMode(display=False) as counter:
        model.patch_expert(patches)
    return int(counter.get_total_flops())


def inspect_visdrone_yaml(yaml_path: str) -> None:
    parsed = load_visdrone_yaml(yaml_path)
    print("VisDrone YAML summary")
    print(f"  root:  {parsed.root}")
    print(f"  train: {summarize_split(parsed.train)}")
    print(f"  val:   {summarize_split(parsed.val)}")
    print(f"  test:  {summarize_split(parsed.test)}")
    print(f"  names: {parsed.names}")
    print()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.visdrone_yaml:
        inspect_visdrone_yaml(args.visdrone_yaml)

    config = H2RConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        max_routes=args.max_routes,
    )
    device = torch.device(args.device)

    model = H2RDetector(config).to(device)
    criterion = H2RLoss(config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_images, train_targets = generate_synthetic_batch(config, args.batch_size)
    train_images = train_images.to(device)
    train_targets = [{k: v.to(device) for k, v in target.items()} for target in train_targets]

    eval_images, eval_targets = generate_synthetic_batch(config, args.batch_size)
    eval_images = eval_images.to(device)
    eval_targets = [{k: v.to(device) for k, v in target.items()} for target in eval_targets]

    model.eval()
    with torch.no_grad():
        initial_train_outputs = model(train_images)
        initial_outputs = model(eval_images)
        initial_train_recall = routing_recall(config, initial_train_outputs["routes"], train_targets)
        initial_recall = routing_recall(config, initial_outputs["routes"], eval_targets)

    print("Initial train routing recall:", f"{initial_train_recall:.3f}")
    print("Initial holdout routing recall:", f"{initial_recall:.3f}")

    model.train()
    for step in range(1, args.steps + 1):
        outputs = model(train_images, teacher_targets=_human_targets(config, train_targets))
        losses = criterion(outputs, train_targets, image_size=(config.image_size, config.image_size))

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 4) == 0:
            print(
                f"step {step:02d} | "
                f"total={losses['total'].item():.4f} "
                f"route={losses['route_loss'].item():.4f} "
                f"refine_obj={losses['refine_obj_loss'].item():.4f} "
                f"budget={losses['budget_loss'].item():.4f}"
            )

    model.eval()
    with torch.no_grad():
        final_train_outputs = model(train_images)
        final_outputs = model(eval_images)
        final_train_recall = routing_recall(config, final_train_outputs["routes"], train_targets)
        final_recall = routing_recall(config, final_outputs["routes"], eval_targets)
        sparse_fraction = area_fraction(final_outputs["routes"], config.image_size)
        sparse_flops = measure_patch_expert_flops(model, final_outputs["patches"])
        dense_patches = model.extract_dense_patches(eval_images)
        dense_flops = measure_patch_expert_flops(model, dense_patches)

    reduction = 0.0 if dense_flops == 0 else 1.0 - sparse_flops / dense_flops
    print()
    print("Post-train overfit routing recall:", f"{final_train_recall:.3f}")
    print("Post-train holdout routing recall:", f"{final_recall:.3f}")
    print("Mean routed area fraction:", f"{sparse_fraction:.3f}")
    print("Sparse refine FLOPs:", sparse_flops)
    print("Dense refine FLOPs:", dense_flops)
    print("Estimated refine FLOP reduction:", f"{reduction:.3%}")
    print("Predicted routes:", final_outputs["routes"].predicted_count)
    print("Teacher routes on final eval:", final_outputs["routes"].teacher_count)


if __name__ == "__main__":
    main()
