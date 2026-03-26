from __future__ import annotations

import random

import torch

from .config import H2RConfig


HUMAN_COLORS = (
    torch.tensor([0.95, 0.85, 0.3]),
    torch.tensor([0.3, 0.85, 0.95]),
)

OTHER_COLORS = (
    torch.tensor([0.95, 0.35, 0.35]),
    torch.tensor([0.35, 0.95, 0.35]),
    torch.tensor([0.55, 0.45, 0.95]),
    torch.tensor([0.9, 0.55, 0.2]),
)


def _draw_box(image: torch.Tensor, x1: int, y1: int, x2: int, y2: int, color: torch.Tensor) -> None:
    image[:, y1:y2, x1:x2] = color[:, None, None]


def _add_noise(image: torch.Tensor) -> None:
    image.add_(0.05 * torch.randn_like(image)).clamp_(0.0, 1.0)


def _random_box(size_range: tuple[int, int], image_size: int) -> tuple[int, int, int, int]:
    w = random.randint(*size_range)
    h = random.randint(*size_range)
    x1 = random.randint(0, max(0, image_size - w - 1))
    y1 = random.randint(0, max(0, image_size - h - 1))
    x2 = min(image_size, x1 + w)
    y2 = min(image_size, y1 + h)
    return x1, y1, x2, y2


def generate_synthetic_batch(config: H2RConfig, batch_size: int) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images = []
    targets: list[dict[str, torch.Tensor]] = []
    image_size = config.image_size

    for _ in range(batch_size):
        image = torch.full((3, image_size, image_size), 0.08)

        for _ in range(random.randint(6, 12)):
            y = random.randint(0, image_size - 1)
            image[:, y : min(image_size, y + 1), :] += torch.rand(3, 1, 1) * 0.02

        boxes = []
        labels = []

        human_count = random.randint(*config.synthetic_people_range)
        for _ in range(human_count):
            x1, y1, x2, y2 = _random_box(config.synthetic_human_size, image_size)
            cls_id = random.choice(config.human_class_ids)
            color = HUMAN_COLORS[cls_id % len(HUMAN_COLORS)]
            _draw_box(image, x1, y1, x2, y2, color)
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)

        other_count = random.randint(*config.synthetic_other_range)
        for _ in range(other_count):
            x1, y1, x2, y2 = _random_box(config.synthetic_other_size, image_size)
            cls_id = random.randint(2, config.num_classes - 1)
            color = OTHER_COLORS[cls_id % len(OTHER_COLORS)]
            _draw_box(image, x1, y1, x2, y2, color)
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)

        _add_noise(image)
        images.append(image.clamp(0.0, 1.0))
        targets.append(
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.long).reshape(-1),
            }
        )

    return torch.stack(images), targets
