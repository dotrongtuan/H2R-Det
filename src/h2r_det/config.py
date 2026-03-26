from __future__ import annotations

from dataclasses import dataclass


DEFAULT_VISDRONE_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


@dataclass(slots=True)
class H2RConfig:
    num_classes: int = 10
    class_names: tuple[str, ...] = DEFAULT_VISDRONE_NAMES
    human_class_ids: tuple[int, int] = (0, 1)
    image_size: int = 640
    patch_size: int = 96
    max_routes: int = 12
    route_stride: int = 4
    route_min_size: int = 32
    route_max_size: int = 160
    route_nms_kernel: int = 5
    route_teacher_pad: float = 1.5
    route_score_power: float = 1.0
    backbone_channels: tuple[int, int, int, int] = (32, 64, 128, 192)
    fpn_channels: int = 96
    refine_channels: tuple[int, int, int] = (64, 96, 128)
    scout_stride: int = 8
    refine_dropout: float = 0.05
    max_humans_per_image: int = 16
    route_positive_radius: int = 2
    route_loss_weight: float = 1.0
    scale_loss_weight: float = 0.25
    scout_cls_weight: float = 1.0
    scout_box_weight: float = 1.0
    refine_obj_weight: float = 1.0
    refine_cls_weight: float = 1.0
    refine_box_weight: float = 1.0
    budget_loss_weight: float = 0.15
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    synthetic_people_range: tuple[int, int] = (3, 9)
    synthetic_other_range: tuple[int, int] = (2, 6)
    synthetic_human_size: tuple[int, int] = (6, 18)
    synthetic_other_size: tuple[int, int] = (10, 48)

    @property
    def human_subclasses(self) -> int:
        return len(self.human_class_ids)
