from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import H2RConfig
from .model import RouteBundle


def _focal_bce(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_factor * (1.0 - p_t).pow(gamma) * ce).mean()


def _gaussian_radius(box_w: float, box_h: float, stride: int) -> int:
    radius = max(box_w, box_h) / max(1.0, 2.0 * stride)
    return max(1, int(round(radius)))


def _draw_gaussian(heatmap: torch.Tensor, center_x: int, center_y: int, radius: int) -> None:
    height, width = heatmap.shape
    x_min = max(0, center_x - radius)
    x_max = min(width - 1, center_x + radius)
    y_min = max(0, center_y - radius)
    y_max = min(height - 1, center_y + radius)
    if x_min > x_max or y_min > y_max:
        return
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            dist_sq = float((x - center_x) ** 2 + (y - center_y) ** 2)
            value = torch.exp(torch.tensor(-dist_sq / max(1.0, radius * radius), device=heatmap.device))
            heatmap[y, x] = torch.maximum(heatmap[y, x], value)


def _fill_window(target: torch.Tensor, mask: torch.Tensor, center_x: int, center_y: int, radius: int, value: torch.Tensor) -> None:
    height = target.shape[-2]
    width = target.shape[-1]
    x_min = max(0, center_x - radius)
    x_max = min(width - 1, center_x + radius)
    y_min = max(0, center_y - radius)
    y_max = min(height - 1, center_y + radius)
    if x_min > x_max or y_min > y_max:
        return
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    target[..., y_min : y_max + 1, x_min : x_max + 1] = value
    mask[..., y_min : y_max + 1, x_min : x_max + 1] = 1.0


def _encode_scout_box(box: torch.Tensor, grid_x: int, grid_y: int, stride: int, image_w: int, image_h: int) -> torch.Tensor:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    offset_x = cx / stride - grid_x
    offset_y = cy / stride - grid_y
    w = (box[2] - box[0]) / image_w
    h = (box[3] - box[1]) / image_h
    return torch.stack([offset_x, offset_y, w, h]).clamp(0.0, 1.0)


def build_router_targets(
    config: H2RConfig,
    targets: list[dict[str, torch.Tensor]],
    feature_shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(targets)
    height, width = feature_shape
    heatmap = torch.zeros((batch_size, 1, height, width), device=device)
    scale_targets = torch.zeros_like(heatmap)
    scale_mask = torch.zeros_like(heatmap)

    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"].to(device)
        labels = target["labels"].to(device)
        if boxes.numel() == 0:
            continue
        human_mask = torch.isin(labels, labels.new_tensor(config.human_class_ids))
        if not human_mask.any():
            continue
        for box in boxes[human_mask]:
            cx = ((box[0] + box[2]) / 2.0) / config.route_stride
            cy = ((box[1] + box[3]) / 2.0) / config.route_stride
            grid_x = int(cx.clamp(0, width - 1).item())
            grid_y = int(cy.clamp(0, height - 1).item())
            radius = max(
                config.route_positive_radius,
                _gaussian_radius(float(box[2] - box[0]), float(box[3] - box[1]), config.route_stride),
            )
            _draw_gaussian(heatmap[batch_idx, 0], grid_x, grid_y, radius)
            side = max(float(box[2] - box[0]), float(box[3] - box[1])) * config.route_teacher_pad
            side = min(max(side, config.route_min_size), config.route_max_size)
            scale_value = heatmap.new_tensor(
                (side - config.route_min_size) / (config.route_max_size - config.route_min_size)
            )
            _fill_window(
                scale_targets[batch_idx : batch_idx + 1],
                scale_mask[batch_idx : batch_idx + 1],
                grid_x,
                grid_y,
                radius=max(0, radius - 1),
                value=scale_value.view(1, 1),
            )

    return heatmap, scale_targets, scale_mask


def build_scout_targets(
    config: H2RConfig,
    targets: list[dict[str, torch.Tensor]],
    feature_shape: tuple[int, int],
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(targets)
    height, width = feature_shape
    image_h, image_w = image_size
    class_targets = torch.zeros((batch_size, config.num_classes, height, width), device=device)
    box_targets = torch.zeros((batch_size, 4, height, width), device=device)
    box_mask = torch.zeros((batch_size, 4, height, width), device=device)

    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"].to(device)
        labels = target["labels"].to(device)
        if boxes.numel() == 0:
            continue
        for box, label in zip(boxes, labels):
            cx = ((box[0] + box[2]) / 2.0) / config.scout_stride
            cy = ((box[1] + box[3]) / 2.0) / config.scout_stride
            grid_x = int(cx.clamp(0, width - 1).item())
            grid_y = int(cy.clamp(0, height - 1).item())
            radius = max(
                config.scout_positive_radius,
                _gaussian_radius(float(box[2] - box[0]), float(box[3] - box[1]), config.scout_stride),
            )
            _draw_gaussian(class_targets[batch_idx, label], grid_x, grid_y, radius)
            box_targets[batch_idx, :, grid_y, grid_x] = _encode_scout_box(
                box,
                grid_x=grid_x,
                grid_y=grid_y,
                stride=config.scout_stride,
                image_w=image_w,
                image_h=image_h,
            )
            box_mask[batch_idx, :, grid_y, grid_x] = 1.0
    return class_targets, box_targets, box_mask


def match_routes_to_humans(
    config: H2RConfig,
    routes: RouteBundle,
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if routes.rois.numel() == 0:
        empty = torch.zeros((0,), device=device)
        return empty[:, None], empty.long(), torch.zeros((0, 4), device=device)

    objectness = torch.zeros((routes.rois.shape[0], 1), device=device)
    subclass_targets = torch.zeros((routes.rois.shape[0],), dtype=torch.long, device=device)
    box_targets = torch.zeros((routes.rois.shape[0], 4), device=device)

    for route_idx, roi in enumerate(routes.rois):
        roi_target = roi.detach()
        batch_idx = int(roi_target[0].item())
        boxes = targets[batch_idx]["boxes"].to(device)
        labels = targets[batch_idx]["labels"].to(device)
        if boxes.numel() == 0:
            continue
        human_mask = torch.isin(labels, labels.new_tensor(config.human_class_ids))
        boxes = boxes[human_mask]
        labels = labels[human_mask]
        if boxes.numel() == 0:
            continue
        centers = torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0), dim=1)
        inside = (
            (centers[:, 0] >= roi_target[1])
            & (centers[:, 0] <= roi_target[3])
            & (centers[:, 1] >= roi_target[2])
            & (centers[:, 1] <= roi_target[4])
        )
        if not inside.any():
            continue
        candidate_boxes = boxes[inside]
        candidate_labels = labels[inside]
        roi_center = torch.stack(((roi_target[1] + roi_target[3]) / 2.0, (roi_target[2] + roi_target[4]) / 2.0))
        candidate_centers = centers[inside]
        distances = torch.norm(candidate_centers - roi_center[None, :], dim=1)
        best_idx = int(torch.argmin(distances).item())
        matched_box = candidate_boxes[best_idx]
        matched_label = int(candidate_labels[best_idx].item())

        inter_x1 = torch.maximum(roi_target[1], matched_box[0])
        inter_y1 = torch.maximum(roi_target[2], matched_box[1])
        inter_x2 = torch.minimum(roi_target[3], matched_box[2])
        inter_y2 = torch.minimum(roi_target[4], matched_box[3])
        inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
        inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
        inter_area = inter_w * inter_h
        box_area = ((matched_box[2] - matched_box[0]) * (matched_box[3] - matched_box[1])).clamp_min(1.0)
        coverage = inter_area / box_area
        if float(coverage.item()) < config.refine_positive_coverage:
            continue

        objectness[route_idx, 0] = 1.0
        subclass_targets[route_idx] = config.human_class_ids.index(matched_label)

        roi_w = max(1.0, float((roi_target[3] - roi_target[1]).item()))
        roi_h = max(1.0, float((roi_target[4] - roi_target[2]).item()))
        cx = ((matched_box[0] + matched_box[2]) / 2.0 - roi_target[1]) / roi_w
        cy = ((matched_box[1] + matched_box[3]) / 2.0 - roi_target[2]) / roi_h
        w = (matched_box[2] - matched_box[0]) / roi_w
        h = (matched_box[3] - matched_box[1]) / roi_h
        box_targets[route_idx] = torch.stack([cx, cy, w, h]).clamp(0.0, 1.0)

    return objectness, subclass_targets, box_targets


class H2RLoss(nn.Module):
    def __init__(self, config: H2RConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: dict[str, object],
        targets: list[dict[str, torch.Tensor]],
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        device = outputs["route_maps"]["human_logits"].device
        route_maps = outputs["route_maps"]
        scout = outputs["scout"]
        routes = outputs["routes"]
        refine = outputs["refine"]

        router_targets, scale_targets, scale_mask = build_router_targets(
            self.config,
            targets,
            feature_shape=route_maps["human_logits"].shape[-2:],
            device=device,
        )
        route_loss = _focal_bce(route_maps["human_logits"], router_targets)
        scale_loss = F.smooth_l1_loss(
            torch.sigmoid(route_maps["scale_logits"]) * scale_mask,
            scale_targets * scale_mask,
            reduction="sum",
        ) / scale_mask.sum().clamp_min(1.0)

        scout_cls_targets, scout_box_targets, scout_box_mask = build_scout_targets(
            self.config,
            targets,
            feature_shape=scout["class_logits"].shape[-2:],
            image_size=image_size,
            device=device,
        )
        scout_cls_loss = _focal_bce(scout["class_logits"], scout_cls_targets)
        scout_box_loss = F.smooth_l1_loss(
            scout["box"] * scout_box_mask,
            scout_box_targets * scout_box_mask,
            reduction="sum",
        ) / scout_box_mask.sum().clamp_min(1.0)

        refine_obj_targets, refine_subclass_targets, refine_box_targets = match_routes_to_humans(
            self.config,
            routes,
            targets,
            device=device,
        )
        if refine_obj_targets.numel():
            refine_obj_loss = F.binary_cross_entropy_with_logits(refine["objectness_logits"], refine_obj_targets)
            positive_mask = refine_obj_targets[:, 0] > 0
            if positive_mask.any():
                refine_cls_loss = F.cross_entropy(
                    refine["class_logits"][positive_mask],
                    refine_subclass_targets[positive_mask],
                )
                refine_box_loss = F.smooth_l1_loss(
                    refine["box"][positive_mask],
                    refine_box_targets[positive_mask],
                )
            else:
                refine_cls_loss = refine["class_logits"].sum() * 0.0
                refine_box_loss = refine["box"].sum() * 0.0
        else:
            refine_obj_loss = refine["objectness_logits"].sum() * 0.0
            refine_cls_loss = refine["class_logits"].sum() * 0.0
            refine_box_loss = refine["box"].sum() * 0.0

        if routes.predicted_rois.numel():
            roi_areas = (routes.predicted_rois[:, 3] - routes.predicted_rois[:, 1]) * (
                routes.predicted_rois[:, 4] - routes.predicted_rois[:, 2]
            )
            budget_loss = (roi_areas / float(image_size[0] * image_size[1])).mean()
        else:
            budget_loss = route_loss.new_tensor(0.0)

        total = (
            self.config.route_loss_weight * route_loss
            + self.config.scale_loss_weight * scale_loss
            + self.config.scout_cls_weight * scout_cls_loss
            + self.config.scout_box_weight * scout_box_loss
            + self.config.refine_obj_weight * refine_obj_loss
            + self.config.refine_cls_weight * refine_cls_loss
            + self.config.refine_box_weight * refine_box_loss
            + self.config.budget_loss_weight * budget_loss
        )
        total = torch.nan_to_num(total, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

        return {
            "total": total,
            "route_loss": route_loss.detach(),
            "scale_loss": scale_loss.detach(),
            "scout_cls_loss": scout_cls_loss.detach(),
            "scout_box_loss": scout_box_loss.detach(),
            "refine_obj_loss": refine_obj_loss.detach(),
            "refine_cls_loss": refine_cls_loss.detach(),
            "refine_box_loss": refine_box_loss.detach(),
            "budget_loss": budget_loss.detach(),
        }
