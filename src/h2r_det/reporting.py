from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from .config import H2RConfig
from .losses import H2RLoss
from .metrics import compute_map50, decode_predictions, mean_routed_area_fraction, routing_recall
from .model import H2RDetector
from .utils import ensure_dir, move_targets_to_device, promote_fp32_tree, write_json
from .visdrone import VisDroneYoloDataset, load_visdrone_yaml


_PLOT_IMPORT_ERROR: Exception | None = None
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - plotting is optional at runtime
    plt = None
    _PLOT_IMPORT_ERROR = exc


_PANEL_TITLES = ("Ground Truth", "Predictions", "Routes")
_COLORS = (
    (231, 76, 60),
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
    (26, 188, 156),
    (149, 165, 166),
    (52, 73, 94),
    (231, 76, 60),
)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _cpu_batch(items: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{key: value.detach().cpu() for key, value in item.items()} for item in items]


def _collate_with_paths(
    batch: list[dict[str, torch.Tensor | str]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[str]]:
    images = torch.stack([item["image"] for item in batch])  # type: ignore[index]
    targets = [{"boxes": item["boxes"], "labels": item["labels"]} for item in batch]  # type: ignore[index]
    image_paths = [str(item["image_path"]) for item in batch]  # type: ignore[index]
    return images, targets, image_paths


def _default_output_dir(checkpoint_path: Path, split: str) -> Path:
    return checkpoint_path.parent / f"{checkpoint_path.stem}_{split}_report"


def _route_subset_for_image(
    route_rois: torch.Tensor,
    route_scores: torch.Tensor,
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if route_rois.numel() == 0:
        return route_rois.new_zeros((0, 4)), route_scores.new_zeros((0,))
    mask = route_rois[:, 0] == float(batch_index)
    return route_rois[mask, 1:].detach().cpu(), route_scores[mask].detach().cpu()


def _ap_from_precision_recall(precision: torch.Tensor, recall: torch.Tensor) -> float:
    if precision.numel() == 0:
        return 0.0
    recall_points = torch.linspace(0.0, 1.0, 101, device=precision.device)
    interpolated = []
    for point in recall_points:
        mask = recall >= point
        interpolated.append(precision[mask].max() if mask.any() else torch.tensor(0.0, device=precision.device))
    return float(torch.stack(interpolated).mean().item())


def _classwise_detection_report(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_names: tuple[str, ...],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    num_classes = len(class_names)
    for class_id in range(num_classes):
        class_predictions: list[tuple[int, float, torch.Tensor]] = []
        gt_by_image: dict[int, torch.Tensor] = {}
        total_gt = 0

        for image_idx, target in enumerate(targets):
            mask = target["labels"] == class_id
            gt_boxes = target["boxes"][mask]
            gt_by_image[image_idx] = gt_boxes
            total_gt += int(gt_boxes.shape[0])

        for image_idx, pred in enumerate(predictions):
            mask = pred["labels"] == class_id
            for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
                class_predictions.append((image_idx, float(score.item()), box))

        class_predictions.sort(key=lambda item: item[1], reverse=True)
        matched = {
            image_idx: torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
            for image_idx, gt_boxes in gt_by_image.items()
        }
        tp: list[float] = []
        fp: list[float] = []

        for image_idx, _, box in class_predictions:
            gt_boxes = gt_by_image[image_idx]
            if gt_boxes.numel() == 0:
                tp.append(0.0)
                fp.append(1.0)
                continue
            ious = box_iou(box[None, :], gt_boxes).squeeze(0)
            best_iou, best_idx = torch.max(ious, dim=0)
            if best_iou >= iou_threshold and not matched[image_idx][best_idx]:
                matched[image_idx][best_idx] = True
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)

        tp_total = int(sum(tp))
        fp_total = int(sum(fp))
        fn_total = max(0, total_gt - tp_total)
        precision_curve = torch.tensor([])
        recall_curve = torch.tensor([])
        ap50 = 0.0
        if tp:
            tp_tensor = torch.tensor(tp)
            fp_tensor = torch.tensor(fp)
            cum_tp = torch.cumsum(tp_tensor, dim=0)
            cum_fp = torch.cumsum(fp_tensor, dim=0)
            precision_curve = cum_tp / (cum_tp + cum_fp).clamp_min(1e-6)
            recall_curve = cum_tp / max(1, total_gt)
            ap50 = _ap_from_precision_recall(precision_curve, recall_curve)

        precision_final = _safe_div(float(tp_total), float(tp_total + fp_total))
        recall_final = _safe_div(float(tp_total), float(total_gt))
        f1 = _safe_div(2.0 * precision_final * recall_final, precision_final + recall_final)
        reports.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "gt_count": total_gt,
                "pred_count": len(class_predictions),
                "tp50": tp_total,
                "fp50": fp_total,
                "fn50": fn_total,
                "precision50": precision_final,
                "recall50": recall_final,
                "f1_50": f1,
                "ap50": ap50,
                "precision_curve": precision_curve.tolist(),
                "recall_curve": recall_curve.tolist(),
            }
        )
    return reports


def _match_detection_stats(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    iou_threshold: float = 0.5,
    label_subset: set[int] | None = None,
) -> dict[str, float]:
    pred_boxes = prediction["boxes"]
    pred_scores = prediction["scores"]
    pred_labels = prediction["labels"]
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]

    if label_subset is not None:
        pred_mask = torch.tensor([int(label.item()) in label_subset for label in pred_labels], dtype=torch.bool)
        gt_mask = torch.tensor([int(label.item()) in label_subset for label in gt_labels], dtype=torch.bool)
        pred_boxes = pred_boxes[pred_mask]
        pred_scores = pred_scores[pred_mask]
        pred_labels = pred_labels[pred_mask]
        gt_boxes = gt_boxes[gt_mask]
        gt_labels = gt_labels[gt_mask]

    pred_order = torch.argsort(pred_scores, descending=True) if pred_scores.numel() else torch.zeros((0,), dtype=torch.long)
    pred_boxes = pred_boxes[pred_order]
    pred_scores = pred_scores[pred_order]
    pred_labels = pred_labels[pred_order]

    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    tp = 0
    fp = 0
    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        candidate_indices = torch.nonzero((gt_labels == pred_label) & (~matched_gt), as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            fp += 1
            continue
        ious = box_iou(pred_box[None, :], gt_boxes[candidate_indices]).squeeze(0)
        best_iou, best_pos = torch.max(ious, dim=0)
        if best_iou >= iou_threshold:
            matched_gt[candidate_indices[best_pos]] = True
            tp += 1
        else:
            fp += 1

    fn = int((~matched_gt).sum().item())
    precision = _safe_div(float(tp), float(tp + fp))
    recall = _safe_div(float(tp), float(tp + fn))
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "gt_count": float(gt_boxes.shape[0]),
        "pred_count": float(pred_boxes.shape[0]),
        "tp50": float(tp),
        "fp50": float(fp),
        "fn50": float(fn),
        "precision50": precision,
        "recall50": recall,
        "f1_50": f1,
    }


def _confusion_matrix(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> list[list[int]]:
    background_idx = num_classes
    matrix = [[0 for _ in range(num_classes + 1)] for _ in range(num_classes + 1)]
    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction["boxes"]
        pred_scores = prediction["scores"]
        pred_labels = prediction["labels"]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        order = torch.argsort(pred_scores, descending=True) if pred_scores.numel() else torch.zeros((0,), dtype=torch.long)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            if gt_boxes.numel() == 0:
                matrix[background_idx][int(pred_label.item())] += 1
                continue
            unmatched = torch.nonzero(~matched_gt, as_tuple=False).squeeze(1)
            if unmatched.numel() == 0:
                matrix[background_idx][int(pred_label.item())] += 1
                continue
            ious = box_iou(pred_box[None, :], gt_boxes[unmatched]).squeeze(0)
            best_iou, best_pos = torch.max(ious, dim=0)
            if best_iou >= iou_threshold:
                gt_index = int(unmatched[best_pos].item())
                matched_gt[gt_index] = True
                matrix[int(gt_labels[gt_index].item())][int(pred_label.item())] += 1
            else:
                matrix[background_idx][int(pred_label.item())] += 1

        for missed_gt_label in gt_labels[~matched_gt]:
            matrix[int(missed_gt_label.item())][background_idx] += 1
    return matrix


def _mean_route_area_fraction_for_image(route_rois: torch.Tensor, image_size: tuple[int, int]) -> float:
    if route_rois.numel() == 0:
        return 0.0
    image_h, image_w = image_size
    areas = (route_rois[:, 2] - route_rois[:, 0]) * (route_rois[:, 3] - route_rois[:, 1])
    return float((areas / float(image_h * image_w)).mean().item())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_training_curves(history_path: Path, output_path: Path) -> str | None:
    if plt is None:
        return f"matplotlib unavailable: {_PLOT_IMPORT_ERROR}"
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload["history"] if isinstance(payload, dict) and "history" in payload else payload
    if not history:
        return "history is empty"
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.plot(epochs, [row.get("train_loss") for row in history], label="train_loss")
    ax.plot(epochs, [row.get("val_loss") for row in history], label="val_loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, [row.get("val_human_ap50") for row in history], label="val_human_ap50")
    ax.plot(epochs, [row.get("val_map50") for row in history], label="val_map50")
    ax.set_title("Validation AP50")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, [row.get("train_route_recall") for row in history], label="train_route_recall")
    ax.plot(epochs, [row.get("val_route_recall") for row in history], label="val_route_recall")
    ax.set_title("Route Recall")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, [row.get("train_routed_area") for row in history], label="train_routed_area")
    ax.plot(epochs, [row.get("val_routed_area") for row in history], label="val_routed_area")
    ax2 = ax.twinx()
    ax2.plot(epochs, [row.get("val_routes_per_image") for row in history], color="tab:red", linestyle="--", label="val_routes_per_image")
    ax.set_title("Routing Budget")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return None


def _plot_per_class_ap50(class_rows: list[dict[str, Any]], output_path: Path) -> str | None:
    if plt is None:
        return f"matplotlib unavailable: {_PLOT_IMPORT_ERROR}"
    labels = [row["class_name"] for row in class_rows]
    scores = [row["ap50"] for row in class_rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(labels)), scores, color="tab:blue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("AP50")
    ax.set_title("Per-class AP50")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return None


def _plot_human_pr_curves(
    class_rows: list[dict[str, Any]],
    human_class_ids: tuple[int, ...],
    output_path: Path,
) -> str | None:
    if plt is None:
        return f"matplotlib unavailable: {_PLOT_IMPORT_ERROR}"
    fig, ax = plt.subplots(figsize=(7, 6))
    for class_id in human_class_ids:
        row = class_rows[class_id]
        recall = row["recall_curve"]
        precision = row["precision_curve"]
        if not recall or not precision:
            continue
        ax.plot(recall, precision, label=f"{row['class_name']} (AP50={row['ap50']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Human-class Precision-Recall")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return None


def _plot_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    output_path: Path,
) -> str | None:
    if plt is None:
        return f"matplotlib unavailable: {_PLOT_IMPORT_ERROR}"
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Detection Confusion Matrix @ IoU 0.5")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return None


def _default_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    return ImageFont.load_default()


def _box_color(label: int) -> tuple[int, int, int]:
    return _COLORS[label % len(_COLORS)]


def _draw_box_panel(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor | None,
    class_names: tuple[str, ...],
    title: str,
) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    font = _default_font()
    draw.rectangle((0, 0, panel.width - 1, 22), fill=(255, 255, 255))
    draw.text((6, 4), title, fill=(0, 0, 0), font=font)
    for index, box in enumerate(boxes.tolist()):
        label = int(labels[index].item())
        color = _box_color(label)
        draw.rectangle(box, outline=color, width=3)
        caption = class_names[label]
        if scores is not None and index < scores.shape[0]:
            caption = f"{caption} {float(scores[index].item()):.2f}"
        text_y = max(24, int(box[1]) - 12)
        draw.rectangle((box[0], text_y, min(panel.width - 1, box[0] + 160), text_y + 12), fill=color)
        draw.text((box[0] + 2, text_y), caption, fill=(255, 255, 255), font=font)
    return panel


def _draw_routes_panel(
    image: Image.Image,
    route_rois: torch.Tensor,
    route_scores: torch.Tensor,
    title: str,
) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    font = _default_font()
    draw.rectangle((0, 0, panel.width - 1, 22), fill=(255, 255, 255))
    draw.text((6, 4), title, fill=(0, 0, 0), font=font)
    for index, box in enumerate(route_rois.tolist()):
        color = (52, 152, 219)
        draw.rectangle(box, outline=color, width=2)
        if index < route_scores.shape[0]:
            caption = f"{float(route_scores[index].item()):.2f}"
            text_y = max(24, int(box[1]) - 12)
            draw.rectangle((box[0], text_y, min(panel.width - 1, box[0] + 60), text_y + 12), fill=color)
            draw.text((box[0] + 2, text_y), caption, fill=(255, 255, 255), font=font)
    return panel


def _resize_image_for_panel(image_path: str, image_size: int) -> Image.Image:
    return Image.open(image_path).convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)


def _select_example_indices(
    per_image_rows: list[dict[str, Any]],
    num_examples: int,
) -> tuple[list[int], list[int]]:
    if num_examples <= 0 or not per_image_rows:
        return [], []
    candidates = [row for row in per_image_rows if row["human_gt_count"] + row["human_pred_count"] > 0]
    if not candidates:
        candidates = per_image_rows

    ranked = sorted(
        candidates,
        key=lambda row: (
            row["human_f1_50"],
            -(row["human_gt_count"] + row["human_pred_count"]),
            row["image_index"],
        ),
    )
    worst_count = min(len(ranked), max(1, math.ceil(num_examples / 2)))
    worst_indices = [int(row["image_index"]) for row in ranked[:worst_count]]

    remaining = num_examples - len(worst_indices)
    best_candidates = list(reversed(ranked))
    best_indices: list[int] = []
    for row in best_candidates:
        image_index = int(row["image_index"])
        if image_index in worst_indices or image_index in best_indices:
            continue
        best_indices.append(image_index)
        if len(best_indices) >= remaining:
            break
    return worst_indices, best_indices


def _save_example_panels(
    output_dir: Path,
    records: list[dict[str, Any]],
    per_image_rows: list[dict[str, Any]],
    class_names: tuple[str, ...],
    image_size: int,
    num_examples: int,
) -> list[str]:
    ensure_dir(output_dir)
    worst_indices, best_indices = _select_example_indices(per_image_rows, num_examples)
    saved_paths: list[str] = []
    selections = [("worst", worst_indices), ("best", best_indices)]
    for prefix, indices in selections:
        for rank, image_index in enumerate(indices, start=1):
            record = records[image_index]
            image = _resize_image_for_panel(record["image_path"], image_size)
            gt_panel = _draw_box_panel(image, record["target"]["boxes"], record["target"]["labels"], None, class_names, _PANEL_TITLES[0])
            pred_panel = _draw_box_panel(
                image,
                record["prediction"]["boxes"],
                record["prediction"]["labels"],
                record["prediction"]["scores"],
                class_names,
                _PANEL_TITLES[1],
            )
            route_panel = _draw_routes_panel(image, record["route_rois"], record["route_scores"], _PANEL_TITLES[2])

            canvas = Image.new("RGB", (image.width * 3, image.height + 28), color=(245, 245, 245))
            canvas.paste(gt_panel, (0, 28))
            canvas.paste(pred_panel, (image.width, 28))
            canvas.paste(route_panel, (image.width * 2, 28))
            title = (
                f"{Path(record['image_path']).name} | "
                f"human_f1={per_image_rows[image_index]['human_f1_50']:.3f} | "
                f"gt={int(per_image_rows[image_index]['human_gt_count'])} | "
                f"pred={int(per_image_rows[image_index]['human_pred_count'])}"
            )
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), title, fill=(0, 0, 0), font=_default_font())

            destination = output_dir / f"{prefix}_{rank:02d}_{Path(record['image_path']).stem}.jpg"
            canvas.save(destination, quality=95)
            saved_paths.append(str(destination))
    return saved_paths


def _report_markdown(
    summary: dict[str, Any],
    class_rows: list[dict[str, Any]],
    artifact_paths: dict[str, str],
) -> str:
    best_classes = sorted(class_rows, key=lambda row: row["ap50"], reverse=True)[:5]
    lines = [
        "# Evaluation Report",
        "",
        "## Summary",
        "",
        f"- checkpoint_epoch: {summary['checkpoint_epoch']}",
        f"- loss: {summary['loss']:.6f}",
        f"- route_recall: {summary['route_recall']:.6f}",
        f"- routed_area: {summary['routed_area']:.6f}",
        f"- routes_per_image: {summary['routes_per_image']:.6f}",
        f"- map50: {summary['map50']:.6f}",
        f"- human_ap50: {summary['human_ap50']:.6f}",
        "",
        "## Top Classes By AP50",
        "",
    ]
    for row in best_classes:
        lines.append(f"- {row['class_name']}: AP50={row['ap50']:.6f}, recall50={row['recall50']:.6f}, precision50={row['precision50']:.6f}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, path in artifact_paths.items():
        lines.append(f"- {name}: {path}")
    lines.append("")
    return "\n".join(lines)


def generate_evaluation_report(
    checkpoint_path: str | Path,
    visdrone_yaml: str | Path,
    dataset_root: str | Path | None = None,
    split: str = "val",
    batch_size: int = 4,
    num_workers: int = 2,
    device: str | torch.device = "cpu",
    limit: int | None = None,
    scout_score_thresh: float = 0.05,
    refine_score_thresh: float = 0.1,
    nms_iou: float = 0.5,
    topk: int = 300,
    output_dir: str | Path | None = None,
    history_path: str | Path | None = None,
    num_examples: int = 12,
    save_json_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir_path = ensure_dir(output_dir or _default_output_dir(checkpoint_path, split))
    plots_dir = ensure_dir(output_dir_path / "plots")
    examples_dir = ensure_dir(output_dir_path / "examples")
    device_obj = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    config = H2RConfig(**checkpoint["config"])

    model = H2RDetector(config).to(device_obj)
    load_result = model.load_state_dict(checkpoint["model"], strict=False)
    criterion = H2RLoss(config)

    parsed = load_visdrone_yaml(visdrone_yaml, override_root=dataset_root)
    split_root = getattr(parsed, split, None)
    if split_root is None:
        raise ValueError(f"Split '{split}' is not defined for {visdrone_yaml}.")

    dataset = VisDroneYoloDataset(split_root, image_size=config.image_size, limit=limit, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=_collate_with_paths,
    )

    model.eval()
    total_loss = 0.0
    total_route_recall = 0.0
    total_area = 0.0
    total_routes_per_image = 0.0
    batches = 0
    predictions_all: list[dict[str, torch.Tensor]] = []
    targets_all: list[dict[str, torch.Tensor]] = []
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for images, targets, image_paths in loader:
            images = images.to(device_obj, non_blocking=True)
            targets = move_targets_to_device(targets, device_obj)
            outputs = model(images)
            outputs_fp32 = promote_fp32_tree(outputs)
            losses = criterion(outputs_fp32, targets, image_size=(config.image_size, config.image_size))
            predictions = decode_predictions(
                outputs_fp32,
                config,
                image_size=(config.image_size, config.image_size),
                scout_score_thresh=scout_score_thresh,
                refine_score_thresh=refine_score_thresh,
                topk=topk,
                nms_iou=nms_iou,
            )

            predictions_cpu = _cpu_batch(predictions)
            targets_cpu = _cpu_batch(targets)
            predicted_rois = outputs["routes"].predicted_rois
            predicted_scores = outputs["routes"].predicted_scores

            for batch_index, image_path in enumerate(image_paths):
                route_rois, route_scores = _route_subset_for_image(predicted_rois, predicted_scores, batch_index)
                records.append(
                    {
                        "image_path": image_path,
                        "prediction": predictions_cpu[batch_index],
                        "target": targets_cpu[batch_index],
                        "route_rois": route_rois,
                        "route_scores": route_scores,
                    }
                )

            predictions_all.extend(predictions_cpu)
            targets_all.extend(targets_cpu)
            total_loss += float(losses["total"].item())
            total_route_recall += routing_recall(config, outputs["routes"], targets)
            total_area += mean_routed_area_fraction(outputs["routes"], (config.image_size, config.image_size))
            total_routes_per_image += outputs["routes"].predicted_count / max(1, images.shape[0])
            batches += 1

    map_metrics = compute_map50(predictions_all, targets_all, config.num_classes)
    class_rows = _classwise_detection_report(predictions_all, targets_all, config.class_names)
    human_ap50 = sum(class_rows[class_id]["ap50"] for class_id in config.human_class_ids) / max(1, len(config.human_class_ids))
    confusion = _confusion_matrix(predictions_all, targets_all, config.num_classes)

    per_image_rows: list[dict[str, Any]] = []
    human_label_subset = set(config.human_class_ids)
    for image_index, record in enumerate(records):
        overall = _match_detection_stats(record["prediction"], record["target"])
        human = _match_detection_stats(record["prediction"], record["target"], label_subset=human_label_subset)
        per_image_rows.append(
            {
                "image_index": image_index,
                "image_path": record["image_path"],
                "route_count": int(record["route_rois"].shape[0]),
                "mean_route_area_fraction": _mean_route_area_fraction_for_image(
                    record["route_rois"],
                    (config.image_size, config.image_size),
                ),
                "overall_gt_count": int(overall["gt_count"]),
                "overall_pred_count": int(overall["pred_count"]),
                "overall_tp50": int(overall["tp50"]),
                "overall_fp50": int(overall["fp50"]),
                "overall_fn50": int(overall["fn50"]),
                "overall_precision50": overall["precision50"],
                "overall_recall50": overall["recall50"],
                "overall_f1_50": overall["f1_50"],
                "human_gt_count": int(human["gt_count"]),
                "human_pred_count": int(human["pred_count"]),
                "human_tp50": int(human["tp50"]),
                "human_fp50": int(human["fp50"]),
                "human_fn50": int(human["fn50"]),
                "human_precision50": human["precision50"],
                "human_recall50": human["recall50"],
                "human_f1_50": human["f1_50"],
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch", -1),
        "split": split,
        "dataset_root": str(parsed.root),
        "num_images": len(dataset),
        "load_adjusted": bool(load_result.missing_keys or load_result.unexpected_keys),
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
        "loss": total_loss / max(1, batches),
        "route_recall": total_route_recall / max(1, batches),
        "routed_area": total_area / max(1, batches),
        "routes_per_image": total_routes_per_image / max(1, batches),
        "map50": float(map_metrics["map50"]),
        "human_ap50": float(human_ap50),
        "human_classes": [config.class_names[class_id] for class_id in config.human_class_ids],
        "ap50_per_class": [row["ap50"] for row in class_rows],
        "best_class_by_ap50": max(class_rows, key=lambda row: row["ap50"])["class_name"] if class_rows else "",
        "output_dir": str(output_dir_path),
    }

    summary_path = output_dir_path / "summary.json"
    write_json(summary_path, summary)
    if save_json_path:
        write_json(save_json_path, summary)

    summary_csv_path = output_dir_path / "summary.csv"
    _write_csv(summary_csv_path, [{"metric": key, "value": value} for key, value in summary.items() if not isinstance(value, (list, dict))])

    class_table_rows = [
        {
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "gt_count": row["gt_count"],
            "pred_count": row["pred_count"],
            "tp50": row["tp50"],
            "fp50": row["fp50"],
            "fn50": row["fn50"],
            "precision50": row["precision50"],
            "recall50": row["recall50"],
            "f1_50": row["f1_50"],
            "ap50": row["ap50"],
        }
        for row in class_rows
    ]
    class_csv_path = output_dir_path / "per_class_metrics.csv"
    class_md_path = output_dir_path / "per_class_metrics.md"
    _write_csv(class_csv_path, class_table_rows)
    _write_markdown_table(class_md_path, class_table_rows)

    per_image_csv_path = output_dir_path / "per_image_metrics.csv"
    _write_csv(per_image_csv_path, per_image_rows)

    confusion_csv_path = output_dir_path / "confusion_matrix.csv"
    confusion_labels = list(config.class_names) + ["background"]
    confusion_rows = []
    for label, row in zip(confusion_labels, confusion):
        confusion_rows.append({"gt_label": label, **{pred_label: value for pred_label, value in zip(confusion_labels, row)}})
    _write_csv(confusion_csv_path, confusion_rows)

    plot_status: dict[str, str] = {}
    history_path_obj = None
    if history_path:
        history_path_obj = Path(history_path)
    else:
        candidate = checkpoint_path.parent / "history.json"
        if candidate.exists():
            history_path_obj = candidate
    if history_path_obj is not None and history_path_obj.exists():
        status = _plot_training_curves(history_path_obj, plots_dir / "training_curves.png")
        if status is not None:
            plot_status["training_curves"] = status

    status = _plot_per_class_ap50(class_rows, plots_dir / "per_class_ap50.png")
    if status is not None:
        plot_status["per_class_ap50"] = status

    status = _plot_human_pr_curves(class_rows, config.human_class_ids, plots_dir / "human_pr_curves.png")
    if status is not None:
        plot_status["human_pr_curves"] = status

    status = _plot_confusion_matrix(confusion, confusion_labels, plots_dir / "confusion_matrix.png")
    if status is not None:
        plot_status["confusion_matrix"] = status

    examples_saved = _save_example_panels(
        examples_dir,
        records,
        per_image_rows,
        config.class_names,
        config.image_size,
        num_examples,
    )

    artifact_paths = {
        "summary_json": str(summary_path),
        "summary_csv": str(summary_csv_path),
        "per_class_csv": str(class_csv_path),
        "per_class_md": str(class_md_path),
        "per_image_csv": str(per_image_csv_path),
        "confusion_csv": str(confusion_csv_path),
        "plots_dir": str(plots_dir),
        "examples_dir": str(examples_dir),
    }
    report_md_path = output_dir_path / "report.md"
    report_md_path.write_text(_report_markdown(summary, class_table_rows, artifact_paths), encoding="utf-8")
    artifact_paths["report_md"] = str(report_md_path)

    if plot_status:
        write_json(output_dir_path / "plot_status.json", plot_status)
        artifact_paths["plot_status_json"] = str(output_dir_path / "plot_status.json")

    archive_path = shutil.make_archive(str(output_dir_path), "zip", root_dir=output_dir_path)

    result = {
        "summary": summary,
        "summary_path": str(summary_path),
        "output_dir": str(output_dir_path),
        "archive_path": archive_path,
        "examples_saved": examples_saved,
        "plot_status": plot_status,
    }
    write_json(output_dir_path / "report_manifest.json", result)
    return result
