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

from h2r_det.reporting import generate_evaluation_report


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
    parser.add_argument("--scout-score-thresh", type=float, default=0.05)
    parser.add_argument("--refine-score-thresh", type=float, default=0.1)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--history-json", type=str, default="")
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = args.output_dir or str(checkpoint_path.parent / f"{checkpoint_path.stem}_{args.split}_report")
    history_json = args.history_json or str(checkpoint_path.parent / "history.json")
    report = generate_evaluation_report(
        checkpoint_path=checkpoint_path,
        visdrone_yaml=args.visdrone_yaml,
        dataset_root=args.dataset_root or None,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        limit=args.limit or None,
        scout_score_thresh=args.scout_score_thresh,
        refine_score_thresh=args.refine_score_thresh,
        nms_iou=args.nms_iou,
        topk=args.topk,
        output_dir=output_dir,
        history_path=history_json if Path(history_json).exists() else None,
        num_examples=args.num_examples,
        save_json_path=args.save_json or None,
    )
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps({"event": "report_written", "output_dir": report["output_dir"], "archive_path": report["archive_path"]}))


if __name__ == "__main__":
    main()
