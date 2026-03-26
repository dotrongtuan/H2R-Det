from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h2r_det import H2RConfig, H2RDetector, H2RLoss, compute_map50, decode_predictions, mean_routed_area_fraction, routing_recall
from h2r_det.utils import move_targets_to_device
from h2r_det.visdrone import build_visdrone_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an H2R-Det checkpoint on VisDrone.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--visdrone-yaml", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scout-score-thresh", type=float, default=0.25)
    parser.add_argument("--refine-score-thresh", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=150)
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def _cpu_batch(items: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{key: value.detach().cpu() for key, value in item.items()} for item in items]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = H2RConfig(**checkpoint["config"])

    model = H2RDetector(config).to(device)
    model.load_state_dict(checkpoint["model"])
    criterion = H2RLoss(config)
    loader = build_visdrone_dataloader(
        args.visdrone_yaml,
        split=args.split,
        image_size=config.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        limit=args.limit or None,
        override_root=args.dataset_root or None,
    )

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
    summary = {
        "loss": total_loss / max(1, batches),
        "route_recall": total_route_recall / max(1, batches),
        "routed_area": total_area / max(1, batches),
        "map50": float(metrics["map50"]),
        "human_ap50": float(human_ap),
        "ap50_per_class": metrics["ap50_per_class"],
        "checkpoint_epoch": checkpoint.get("epoch", -1),
    }
    print(json.dumps(summary, indent=2))

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
