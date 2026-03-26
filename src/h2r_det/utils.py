from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import H2RConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_targets_to_device(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def human_only_targets(config: H2RConfig, targets: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    human_targets: list[dict[str, torch.Tensor]] = []
    for target in targets:
        boxes = target["boxes"]
        labels = target["labels"]
        if boxes.numel() == 0:
            human_targets.append({"boxes": boxes, "labels": labels})
            continue
        human_mask = torch.isin(labels, labels.new_tensor(config.human_class_ids))
        human_targets.append({"boxes": boxes[human_mask], "labels": labels[human_mask]})
    return human_targets


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: H2RConfig,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload
