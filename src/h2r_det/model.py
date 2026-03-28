from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from .config import H2RConfig


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyBackbone(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int]):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = ConvBNAct(3, c1, 3, stride=2)
        self.stage2 = nn.Sequential(DepthwiseBlock(c1, c2, stride=2), DepthwiseBlock(c2, c2))
        self.stage3 = nn.Sequential(DepthwiseBlock(c2, c3, stride=2), DepthwiseBlock(c3, c3))
        self.stage4 = nn.Sequential(DepthwiseBlock(c3, c4, stride=2), DepthwiseBlock(c4, c4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        p2 = self.stage2(x)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        return p2, p3, p4


class LightweightFPN(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int], out_channels: int):
        super().__init__()
        self.lateral2 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral4 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.out2 = ConvBNAct(out_channels, out_channels, 3)
        self.out3 = ConvBNAct(out_channels, out_channels, 3)

    def forward(self, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p4_lat = self.lateral4(p4)
        p3_lat = self.lateral3(p3) + F.interpolate(p4_lat, size=p3.shape[-2:], mode="nearest")
        p2_lat = self.lateral2(p2) + F.interpolate(p3_lat, size=p2.shape[-2:], mode="nearest")
        return self.out2(p2_lat), self.out3(p3_lat)


class RouteHead(nn.Module):
    def __init__(self, channels: int, use_uncertainty: bool = True):
        super().__init__()
        self.pre = ConvBNAct(channels, channels, 3)
        self.logits = nn.Conv2d(channels, 1, 1)
        self.scale = nn.Conv2d(channels, 1, 1)
        self.uncertainty = nn.Conv2d(channels, 1, 1) if use_uncertainty else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.pre(x)
        if self.uncertainty is None:
            uncertainty_logits = hidden.new_zeros((hidden.shape[0], 1, hidden.shape[2], hidden.shape[3]))
        else:
            uncertainty_logits = self.uncertainty(hidden)
        return {
            "human_logits": self.logits(hidden),
            "scale_logits": self.scale(hidden),
            "uncertainty_logits": uncertainty_logits,
        }


class ScoutHead(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.stem = ConvBNAct(channels, channels, 3)
        self.class_logits = nn.Conv2d(channels, num_classes, 1)
        self.box = nn.Conv2d(channels, 4, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.stem(x)
        return {
            "class_logits": self.class_logits(hidden),
            "box": torch.sigmoid(self.box(hidden)),
        }


class PatchExpert(nn.Module):
    def __init__(self, in_channels: int, channels: tuple[int, int, int], human_subclasses: int, dropout: float):
        super().__init__()
        c1, c2, c3 = channels
        self.encoder = nn.Sequential(
            DepthwiseBlock(in_channels, c1, stride=2),
            DepthwiseBlock(c1, c2, stride=2),
            DepthwiseBlock(c2, c3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.objectness = nn.Linear(c3, 1)
        self.classifier = nn.Linear(c3, human_subclasses)
        self.box = nn.Linear(c3, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.numel() == 0:
            subclasses = self.classifier.out_features
            return {
                "objectness_logits": x.new_zeros((0, 1)),
                "class_logits": x.new_zeros((0, subclasses)),
                "box": x.new_zeros((0, 4)),
            }
        features = self.encoder(x)
        pooled = self.pool(features).flatten(1)
        pooled = self.dropout(pooled)
        return {
            "objectness_logits": self.objectness(pooled),
            "class_logits": self.classifier(pooled),
            "box": torch.sigmoid(self.box(pooled)),
        }


@dataclass
class RouteBundle:
    rois: torch.Tensor
    scores: torch.Tensor
    predicted_count: int
    teacher_count: int
    per_image_counts: list[int]

    @property
    def predicted_rois(self) -> torch.Tensor:
        return self.rois[: self.predicted_count]

    @property
    def predicted_scores(self) -> torch.Tensor:
        return self.scores[: self.predicted_count]


class HumanAwareRouter(nn.Module):
    def __init__(self, config: H2RConfig):
        super().__init__()
        self.config = config

    def _local_max_mask(self, scores: torch.Tensor) -> torch.Tensor:
        kernel = self.config.route_nms_kernel
        pooled = F.max_pool2d(scores, kernel_size=kernel, stride=1, padding=kernel // 2)
        return scores.eq(pooled)

    def _build_rois(
        self,
        score_map: torch.Tensor,
        scale_map: torch.Tensor,
        uncertainty_map: torch.Tensor,
        image_size: tuple[int, int],
    ) -> RouteBundle:
        batch_size, _, _, width = score_map.shape
        image_h, image_w = image_size
        all_rois = []
        all_scores = []
        per_image_counts: list[int] = []

        for batch_idx in range(batch_size):
            scores = score_map[batch_idx : batch_idx + 1]
            mask = self._local_max_mask(scores)
            filtered = scores.masked_fill(~mask, -1.0)
            topk = min(self.config.max_routes, filtered.numel())
            values, indices = torch.topk(filtered.view(-1), k=topk)
            valid = values > 0
            values = values[valid]
            indices = indices[valid]
            count = int(values.numel())
            per_image_counts.append(count)
            if count == 0:
                continue

            ys = torch.div(indices, width, rounding_mode="floor")
            xs = indices % width
            cx = (xs.float() + 0.5) * self.config.route_stride
            cy = (ys.float() + 0.5) * self.config.route_stride

            scale_logits = scale_map[batch_idx, 0, ys, xs]
            uncertainty = torch.sigmoid(uncertainty_map[batch_idx, 0, ys, xs])
            side = self.config.route_min_size + torch.sigmoid(scale_logits) * (
                self.config.route_max_size - self.config.route_min_size
            )
            side = side * (1.0 + 0.25 * uncertainty)
            half = side / 2.0
            x1 = (cx - half).clamp(0.0, image_w - 1.0)
            y1 = (cy - half).clamp(0.0, image_h - 1.0)
            x2 = (cx + half).clamp(1.0, image_w * 1.0)
            y2 = (cy + half).clamp(1.0, image_h * 1.0)

            rois = torch.stack(
                [
                    torch.full_like(x1, float(batch_idx)),
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                dim=1,
            )
            all_rois.append(rois)
            all_scores.append(values)

        if not all_rois:
            empty = score_map.new_zeros((0, 5))
            return RouteBundle(
                rois=empty,
                scores=score_map.new_zeros((0,)),
                predicted_count=0,
                teacher_count=0,
                per_image_counts=per_image_counts,
            )

        return RouteBundle(
            rois=torch.cat(all_rois, dim=0),
            scores=torch.cat(all_scores, dim=0),
            predicted_count=sum(per_image_counts),
            teacher_count=0,
            per_image_counts=per_image_counts,
        )

    def _teacher_rois(self, targets: list[dict[str, torch.Tensor]], image_size: tuple[int, int]) -> torch.Tensor:
        image_h, image_w = image_size
        teacher_rois = []
        for batch_idx, target in enumerate(targets):
            if target["boxes"].numel() == 0:
                continue
            labels = target["labels"]
            human_mask = torch.isin(labels, labels.new_tensor(self.config.human_class_ids))
            if not human_mask.any():
                continue
            boxes = target["boxes"][human_mask]
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            side = torch.maximum(w, h) * self.config.route_teacher_pad
            side = side.clamp(self.config.route_min_size, self.config.route_max_size)
            half = side / 2.0
            x1 = (cx - half).clamp(0.0, image_w - 1.0)
            y1 = (cy - half).clamp(0.0, image_h - 1.0)
            x2 = (cx + half).clamp(1.0, image_w * 1.0)
            y2 = (cy + half).clamp(1.0, image_h * 1.0)
            rois = torch.stack(
                [
                    torch.full_like(x1, float(batch_idx)),
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                dim=1,
            )
            teacher_rois.append(rois)
        if not teacher_rois:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.cat(teacher_rois, dim=0)

    def forward(
        self,
        route_logits: torch.Tensor,
        scale_logits: torch.Tensor,
        uncertainty_logits: torch.Tensor,
        image_size: tuple[int, int],
        teacher_targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> RouteBundle:
        score_map = torch.sigmoid(route_logits).pow(self.config.route_score_power)
        if self.config.use_route_uncertainty:
            score_map = score_map * (1.0 - 0.35 * torch.sigmoid(uncertainty_logits))
            roi_uncertainty = uncertainty_logits
        else:
            roi_uncertainty = torch.zeros_like(uncertainty_logits)
        bundle = self._build_rois(score_map, scale_logits, roi_uncertainty, image_size)
        rois = bundle.rois
        scores = bundle.scores

        teacher_count = 0
        if teacher_targets is not None:
            teacher_rois = self._teacher_rois(teacher_targets, image_size).to(route_logits.device)
            teacher_count = int(teacher_rois.shape[0])
            if teacher_count:
                rois = torch.cat([rois, teacher_rois], dim=0) if rois.numel() else teacher_rois
                teacher_scores = route_logits.new_full((teacher_count,), 1.0)
                scores = torch.cat([scores, teacher_scores], dim=0) if scores.numel() else teacher_scores

        return RouteBundle(
            rois=rois,
            scores=scores,
            predicted_count=bundle.predicted_count,
            teacher_count=teacher_count,
            per_image_counts=bundle.per_image_counts,
        )


class H2RDetector(nn.Module):
    def __init__(self, config: H2RConfig):
        super().__init__()
        self.config = config
        self.backbone = TinyBackbone(config.backbone_channels)
        self.fpn = LightweightFPN(config.backbone_channels[1:], config.fpn_channels)
        self.route_head = RouteHead(config.fpn_channels, use_uncertainty=config.use_route_uncertainty)
        self.scout_head = ScoutHead(config.fpn_channels, config.num_classes)
        self.router = HumanAwareRouter(config)
        self.patch_expert = PatchExpert(3, config.refine_channels, config.human_subclasses, config.refine_dropout)

    def forward(
        self,
        images: torch.Tensor,
        teacher_targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, object]:
        image_h, image_w = images.shape[-2:]
        p2, p3, p4 = self.backbone(images)
        route_features, scout_features = self.fpn(p2, p3, p4)

        route_outputs = self.route_head(route_features)
        scout_outputs = self.scout_head(scout_features)

        routes = self.router(
            route_outputs["human_logits"],
            route_outputs["scale_logits"],
            route_outputs["uncertainty_logits"],
            image_size=(image_h, image_w),
            teacher_targets=teacher_targets,
        )

        if routes.rois.numel():
            patches = roi_align(
                images,
                routes.rois,
                output_size=(self.config.patch_size, self.config.patch_size),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )
        else:
            patches = images.new_zeros((0, 3, self.config.patch_size, self.config.patch_size))

        refine_outputs = self.patch_expert(patches)
        return {
            "route_maps": route_outputs,
            "scout": scout_outputs,
            "routes": routes,
            "patches": patches,
            "refine": refine_outputs,
        }

    def extract_dense_patches(self, images: torch.Tensor, window_size: int | None = None, stride: int | None = None) -> torch.Tensor:
        image_h, image_w = images.shape[-2:]
        side = int(window_size or max(self.config.route_min_size, self.config.patch_size))
        step = int(stride or max(8, side // 2))
        rois = []
        for batch_idx in range(images.shape[0]):
            y_positions = list(range(0, max(1, image_h - side + 1), step))
            x_positions = list(range(0, max(1, image_w - side + 1), step))
            if y_positions[-1] != image_h - side:
                y_positions.append(max(0, image_h - side))
            if x_positions[-1] != image_w - side:
                x_positions.append(max(0, image_w - side))
            for y1 in y_positions:
                for x1 in x_positions:
                    x2 = min(image_w, x1 + side)
                    y2 = min(image_h, y1 + side)
                    rois.append([float(batch_idx), float(x1), float(y1), float(x2), float(y2)])
        if not rois:
            return images.new_zeros((0, 3, self.config.patch_size, self.config.patch_size))
        rois_tensor = torch.tensor(rois, device=images.device, dtype=images.dtype)
        return roi_align(
            images,
            rois_tensor,
            output_size=(self.config.patch_size, self.config.patch_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
