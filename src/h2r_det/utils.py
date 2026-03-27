from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

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


def promote_fp32_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: promote_fp32_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [promote_fp32_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(promote_fp32_tree(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        kwargs = {field.name: promote_fp32_tree(getattr(value, field.name)) for field in fields(value)}
        return type(value)(**kwargs)
    return value


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source_state = model.state_dict()
        target_state = self.ema.state_dict()
        for key, target_value in target_state.items():
            source_value = source_state[key].detach()
            if not source_value.is_floating_point():
                target_value.copy_(source_value)
            else:
                target_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)

    def to(self, device: torch.device) -> "ModelEMA":
        self.ema.to(device)
        return self


def get_env_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_env_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return get_env_world_size() > 1


def is_main_process() -> bool:
    return get_env_rank() == 0


def init_distributed(device_preference: str) -> tuple[torch.device, int, int, int]:
    world_size = get_env_world_size()
    rank = get_env_rank()
    local_rank = get_env_local_rank()

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and "cuda" in device_preference else "gloo"
        kwargs: dict[str, Any] = {}
        if backend == "nccl":
            kwargs["device_id"] = local_rank
        dist.init_process_group(backend=backend, **kwargs)

    if world_size > 1 and torch.cuda.is_available() and "cuda" in device_preference:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_preference)

    return device, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
