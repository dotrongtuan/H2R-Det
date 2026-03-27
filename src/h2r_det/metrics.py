from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import batched_nms, box_iou

from .config import H2RConfig
from .model import RouteBundle


def _decode_scout_boxes(
    boxes: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    image_size: tuple[int, int],
    stride: int,
) -> torch.Tensor:
    image_h, image_w = image_size
    cx = (xs.float() + boxes[:, 0]) * stride
    cy = (ys.float() + boxes[:, 1]) * stride
    w = boxes[:, 2] * image_w
    h = boxes[:, 3] * image_h
    x1 = (cx - w / 2.0).clamp(0.0, image_w - 1.0)
    y1 = (cy - h / 2.0).clamp(0.0, image_h - 1.0)
    x2 = (cx + w / 2.0).clamp(1.0, image_w * 1.0)
    y2 = (cy + h / 2.0).clamp(1.0, image_h * 1.0)
    return torch.stack([x1, y1, x2, y2], dim=1)


def _roi_relative_cxcywh_to_xyxy(rel_boxes: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
    if rel_boxes.numel() == 0:
        return rel_boxes.new_zeros((0, 4))
    roi_w = (rois[:, 2] - rois[:, 0]).clamp_min(1.0)
    roi_h = (rois[:, 3] - rois[:, 1]).clamp_min(1.0)
    cx = rois[:, 0] + rel_boxes[:, 0] * roi_w
    cy = rois[:, 1] + rel_boxes[:, 1] * roi_h
    w = rel_boxes[:, 2] * roi_w
    h = rel_boxes[:, 3] * roi_h
    x1 = (cx - w / 2.0).clamp(min=0.0)
    y1 = (cy - h / 2.0).clamp(min=0.0)
    x2 = (cx + w / 2.0).clamp(min=1.0)
    y2 = (cy + h / 2.0).clamp(min=1.0)
    return torch.stack([x1, y1, x2, y2], dim=1)


def predicted_route_subset(routes: RouteBundle) -> RouteBundle:
    return RouteBundle(
        rois=routes.predicted_rois,
        scores=routes.predicted_scores,
        predicted_count=routes.predicted_count,
        teacher_count=0,
        per_image_counts=routes.per_image_counts,
    )


def _local_peak_mask(scores: torch.Tensor, kernel: int) -> torch.Tensor:
    pooled = torch.nn.functional.max_pool2d(scores, kernel_size=kernel, stride=1, padding=kernel // 2)
    return scores.eq(pooled)


def routing_recall(config: H2RConfig, routes: RouteBundle, targets: list[dict[str, torch.Tensor]]) -> float:
    routes = predicted_route_subset(routes)
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


def mean_routed_area_fraction(routes: RouteBundle, image_size: tuple[int, int]) -> float:
    routes = predicted_route_subset(routes)
    if routes.rois.numel() == 0:
        return 0.0
    image_h, image_w = image_size
    areas = (routes.rois[:, 3] - routes.rois[:, 1]) * (routes.rois[:, 4] - routes.rois[:, 2])
    return float((areas / float(image_h * image_w)).mean().item())


def decode_predictions(
    outputs: dict[str, object],
    config: H2RConfig,
    image_size: tuple[int, int],
    scout_score_thresh: float = 0.25,
    refine_score_thresh: float = 0.25,
    topk: int = 150,
    nms_iou: float = 0.5,
) -> list[dict[str, torch.Tensor]]:
    scout = outputs["scout"]
    routes = outputs["routes"]
    refine = outputs["refine"]
    class_logits = scout["class_logits"]
    scout_boxes = scout["box"]
    batch_size, num_classes, height, width = class_logits.shape
    predictions: list[dict[str, torch.Tensor]] = []

    human_rois = routes.rois
    human_objectness = torch.sigmoid(refine["objectness_logits"]).squeeze(1) if refine["objectness_logits"].numel() else None
    human_probs = torch.softmax(refine["class_logits"], dim=1) if refine["class_logits"].numel() else None
    human_route_scores = routes.scores if routes.scores.numel() else None
    human_ids = torch.tensor(config.human_class_ids, device=class_logits.device, dtype=torch.long)

    for batch_idx in range(batch_size):
        class_probs = torch.sigmoid(class_logits[batch_idx])
        best_scores, best_labels = class_probs.max(dim=0)
        peak_mask = _local_peak_mask(best_scores[None, None], config.scout_decode_kernel).squeeze(0).squeeze(0)
        filtered_scores = best_scores.masked_fill(~peak_mask, -1.0)
        candidate_k = min(topk, filtered_scores.numel())
        top_scores, top_indices = torch.topk(filtered_scores.flatten(), k=candidate_k)
        label_grid = best_labels.flatten()
        candidate_labels = label_grid[top_indices]
        human_mask = torch.isin(candidate_labels, human_ids)
        keep = top_scores >= scout_score_thresh
        keep = keep & (~human_mask | (top_scores >= config.human_scout_score_thresh))
        top_scores = top_scores[keep]
        top_indices = top_indices[keep]
        candidate_labels = candidate_labels[keep]
        scout_det_boxes = class_logits.new_zeros((0, 4))
        scout_det_scores = class_logits.new_zeros((0,))
        scout_det_labels = class_logits.new_zeros((0,), dtype=torch.long)
        if top_scores.numel():
            ys = torch.div(top_indices, width, rounding_mode="floor")
            xs = top_indices % width
            rel_boxes = scout_boxes[batch_idx, :, ys, xs].permute(1, 0)
            scout_det_boxes = _decode_scout_boxes(rel_boxes, xs, ys, image_size, stride=config.scout_stride)
            scout_det_scores = top_scores
            scout_det_labels = candidate_labels.long()

        refine_det_boxes = class_logits.new_zeros((0, 4))
        refine_det_scores = class_logits.new_zeros((0,))
        refine_det_labels = class_logits.new_zeros((0,), dtype=torch.long)
        if human_rois.numel():
            route_mask = human_rois[:, 0] == float(batch_idx)
            if route_mask.any():
                route_indices = torch.nonzero(route_mask, as_tuple=False).squeeze(1)
                obj = human_objectness[route_indices]
                probs = human_probs[route_indices]
                route_scores = human_route_scores[route_indices].pow(config.refine_route_score_power)
                conf, subclass_idx = (obj[:, None] * probs * route_scores[:, None]).max(dim=1)
                keep_refine = conf >= refine_score_thresh
                if keep_refine.any():
                    kept_indices = route_indices[keep_refine]
                    refine_det_boxes = _roi_relative_cxcywh_to_xyxy(refine["box"][kept_indices], human_rois[kept_indices, 1:])
                    refine_det_scores = conf[keep_refine]
                    refine_det_labels = human_ids[subclass_idx[keep_refine]]

        all_boxes = torch.cat([scout_det_boxes, refine_det_boxes], dim=0)
        all_scores = torch.cat([scout_det_scores, refine_det_scores], dim=0)
        all_labels = torch.cat([scout_det_labels, refine_det_labels], dim=0)
        if all_boxes.numel():
            nms_labels = all_labels.clone()
            human_label_mask = torch.isin(nms_labels, human_ids)
            if human_label_mask.any():
                nms_labels[human_label_mask] = human_ids[0]
            keep_idx = batched_nms(all_boxes, all_scores, nms_labels, nms_iou)
            all_boxes = all_boxes[keep_idx]
            all_scores = all_scores[keep_idx]
            all_labels = all_labels[keep_idx]
        predictions.append({"boxes": all_boxes, "scores": all_scores, "labels": all_labels})
    return predictions


def _ap_from_precision_recall(precision: torch.Tensor, recall: torch.Tensor) -> float:
    if precision.numel() == 0:
        return 0.0
    recall_points = torch.linspace(0.0, 1.0, 101, device=precision.device)
    interpolated = []
    for point in recall_points:
        mask = recall >= point
        interpolated.append(precision[mask].max() if mask.any() else torch.tensor(0.0, device=precision.device))
    return float(torch.stack(interpolated).mean().item())


def compute_map50(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    ap_per_class: list[float] = []
    for class_id in range(num_classes):
        class_predictions = []
        gt_by_image: dict[int, torch.Tensor] = {}
        total_gt = 0
        for image_idx, target in enumerate(targets):
            labels = target["labels"]
            boxes = target["boxes"]
            mask = labels == class_id
            gt_boxes = boxes[mask]
            gt_by_image[image_idx] = gt_boxes
            total_gt += int(gt_boxes.shape[0])

        for image_idx, pred in enumerate(predictions):
            labels = pred["labels"]
            boxes = pred["boxes"]
            scores = pred["scores"]
            mask = labels == class_id
            for box, score in zip(boxes[mask], scores[mask]):
                class_predictions.append((image_idx, float(score.item()), box))

        if total_gt == 0:
            ap_per_class.append(0.0)
            continue

        class_predictions.sort(key=lambda item: item[1], reverse=True)
        tp = []
        fp = []
        matched: dict[int, torch.Tensor] = {
            image_idx: torch.zeros(gt.shape[0], dtype=torch.bool, device=gt.device) for image_idx, gt in gt_by_image.items()
        }

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

        if not tp:
            ap_per_class.append(0.0)
            continue

        tp_tensor = torch.tensor(tp)
        fp_tensor = torch.tensor(fp)
        cum_tp = torch.cumsum(tp_tensor, dim=0)
        cum_fp = torch.cumsum(fp_tensor, dim=0)
        precision = cum_tp / (cum_tp + cum_fp).clamp_min(1e-6)
        recall = cum_tp / max(1, total_gt)
        ap_per_class.append(_ap_from_precision_recall(precision, recall))

    map50 = sum(ap_per_class) / max(1, num_classes)
    return {"map50": map50, "ap50_per_class": ap_per_class}
