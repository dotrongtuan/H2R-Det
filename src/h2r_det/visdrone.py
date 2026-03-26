from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms.functional import pil_to_tensor

from .config import DEFAULT_VISDRONE_NAMES


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
COMMON_SPLIT_CANDIDATES = {
    "train": (
        "train/images",
        "train",
        "VisDrone2019-DET-train/images",
        "VisDrone2019-DET-train",
    ),
    "val": (
        "val/images",
        "val",
        "VisDrone2019-DET-val/images",
        "VisDrone2019-DET-val",
    ),
    "test": (
        "test/images",
        "test",
        "VisDrone2019-DET-test-dev/images",
        "VisDrone2019-DET-test-dev",
        "VisDrone2019-DET-test-challenge/images",
        "VisDrone2019-DET-test-challenge",
    ),
}


@dataclass(slots=True)
class VisDroneYaml:
    root: Path
    train: Path | None
    val: Path | None
    test: Path | None
    names: tuple[str, ...]
    nc: int


def _resolve_root(root_candidate: str | Path, anchor: Path) -> Path:
    root = Path(root_candidate).expanduser()
    if not root.is_absolute():
        root = (anchor / root).resolve()
    return root


def _resolve_split_entry(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    entry = Path(value)
    if not entry.is_absolute():
        entry = (root / entry).resolve()
    return entry


def _infer_split(root: Path, split: str) -> Path | None:
    for candidate in COMMON_SPLIT_CANDIDATES[split]:
        path = (root / candidate).resolve()
        if path.exists():
            return path
    return None


def _kaggle_input_hint() -> str:
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        children = sorted(path.name for path in kaggle_input.iterdir())
        if children:
            return f"Available /kaggle/input datasets: {', '.join(children[:12])}"
        return "/kaggle/input exists but appears empty."
    return ""


def infer_visdrone_layout(root: str | Path) -> VisDroneYaml:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        hint = _kaggle_input_hint()
        suffix = f" {hint}" if hint else ""
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}.{suffix}")
    train = _infer_split(root_path, "train")
    val = _infer_split(root_path, "val")
    test = _infer_split(root_path, "test")
    if train is None and val is None and test is None:
        hint = _kaggle_input_hint()
        suffix = f" {hint}" if hint else ""
        raise FileNotFoundError(
            "Could not infer a VisDrone layout under "
            f"{root_path}. Expected folders such as 'train/images', 'val/images', "
            "'VisDrone2019-DET-train/images', or 'VisDrone2019-DET-val/images'." + suffix
        )
    return VisDroneYaml(
        root=root_path,
        train=train,
        val=val,
        test=test,
        names=DEFAULT_VISDRONE_NAMES,
        nc=len(DEFAULT_VISDRONE_NAMES),
    )


def load_visdrone_yaml(path: str | Path, override_root: str | Path | None = None) -> VisDroneYaml:
    path_obj = Path(path).expanduser()
    yaml_path = path_obj.resolve() if path_obj.exists() else path_obj

    if yaml_path.exists() and yaml_path.is_dir():
        return infer_visdrone_layout(yaml_path)

    if yaml_path.exists() and yaml_path.is_file():
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        root_candidate = override_root if override_root is not None else payload.get("path", yaml_path.parent)
        root = _resolve_root(root_candidate, yaml_path.parent)

        names = payload.get("names", {})
        if isinstance(names, dict):
            ordered_names = tuple(value for _, value in sorted(names.items(), key=lambda item: int(item[0])))
        else:
            ordered_names = tuple(names)

        return VisDroneYaml(
            root=root,
            train=_resolve_split_entry(root, payload.get("train")),
            val=_resolve_split_entry(root, payload.get("val")),
            test=_resolve_split_entry(root, payload.get("test")),
            names=ordered_names,
            nc=int(payload.get("nc", len(ordered_names))),
        )

    if override_root is not None:
        return infer_visdrone_layout(override_root)

    hint = _kaggle_input_hint()
    suffix = f" {hint}" if hint else ""
    raise FileNotFoundError(
        f"VisDrone YAML or dataset directory was not found: {path_obj}. "
        "Pass a valid YAML file, or pass the dataset directory itself, or set --dataset-root." + suffix
    )


def _scan_images(entry: Path) -> list[Path]:
    if entry.is_file() and entry.suffix.lower() == ".txt":
        lines = [line.strip() for line in entry.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [Path(line).expanduser().resolve() for line in lines]
    if entry.is_dir():
        return sorted(path for path in entry.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return []


def _default_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


class VisDroneYoloDataset(Dataset):
    def __init__(self, images_root: str | Path, image_size: int | None = None, limit: int | None = None):
        self.images_root = Path(images_root).expanduser().resolve()
        self.images = _scan_images(self.images_root)
        if limit is not None:
            self.images = self.images[:limit]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = pil_to_tensor(image).float() / 255.0
        label_path = _default_label_path(image_path)
        boxes = []
        labels = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = parts
                cls_id = int(float(cls))
                cx = float(cx) * image.width
                cy = float(cy) * image.height
                w = float(w) * image.width
                h = float(h) * image.height
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.long).reshape(-1)
        if self.image_size is not None:
            original_h, original_w = image_tensor.shape[-2:]
            resized = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            image_tensor = pil_to_tensor(resized).float() / 255.0
            if boxes_tensor.numel():
                scale_x = self.image_size / original_w
                scale_y = self.image_size / original_h
                boxes_tensor[:, [0, 2]] *= scale_x
                boxes_tensor[:, [1, 3]] *= scale_y

        return {
            "image": image_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_path": str(image_path),
        }


def summarize_split(entry: Path | None) -> str:
    if entry is None:
        return "missing"
    images = _scan_images(entry)
    return f"{len(images)} images from {entry}"


def collate_visdrone(batch: list[dict[str, torch.Tensor | str]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images = torch.stack([item["image"] for item in batch])  # type: ignore[index]
    targets = [{"boxes": item["boxes"], "labels": item["labels"]} for item in batch]  # type: ignore[index]
    return images, targets


def build_visdrone_dataloader(
    yaml_path: str | Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    limit: int | None = None,
    override_root: str | Path | None = None,
    distributed: bool = False,
    drop_last: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    parsed = load_visdrone_yaml(yaml_path, override_root=override_root)
    split_root = getattr(parsed, split, None)
    if split_root is None:
        raise ValueError(f"Split '{split}' is not defined in {yaml_path}.")
    dataset = VisDroneYoloDataset(split_root, image_size=image_size, limit=limit)
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        collate_fn=collate_visdrone,
    )
    return loader, sampler
