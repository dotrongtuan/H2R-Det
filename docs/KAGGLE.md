# Kaggle Guide

This guide assumes:

- your GitHub repo contains this codebase
- your VisDrone dataset is available in Kaggle as an input dataset
- your YAML file either already points at the right root, or you override it with `--dataset-root`

## 1. Create A Kaggle Notebook

- Accelerator: `GPU` if available
- Internet: optional if you clone from GitHub directly

## 2. Add Inputs

Add:

- your code repo, either by `git clone` or by uploading a zip
- your VisDrone dataset as a Kaggle Dataset input

Typical Kaggle paths:

- code repo clone: `/kaggle/working/h2r-det`
- dataset input: `/kaggle/input/visdrone-dataset`

## 3. Bootstrap The Environment

In a Kaggle notebook cell:

```bash
!git clone https://github.com/<your-username>/<your-repo>.git /kaggle/working/h2r-det
%cd /kaggle/working/h2r-det
!python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
!python -m pip install -e . --no-deps
```

If you already uploaded the repo as a dataset instead of cloning, just `cd` into it.

If your Kaggle image is missing one of the dependencies, fall back to:

```bash
!python -m pip install -r requirements.txt
```

## 4. Check The YAML Wiring

Run:

```bash
!python scripts/sanity_check.py --visdrone-yaml /kaggle/input/visdrone-dataset/VisDrone.yaml --steps 1
```

If your YAML uses a different `path:` root than Kaggle's mounted input path, keep the YAML file but override the root during training with `--dataset-root`.

## 5. Fast Smoke Test On A Small Subset

```bash
!python scripts/train_visdrone.py \
  --visdrone-yaml /kaggle/input/visdrone-dataset/VisDrone.yaml \
  --dataset-root /kaggle/input/visdrone-dataset \
  --epochs 1 \
  --batch-size 4 \
  --image-size 512 \
  --patch-size 80 \
  --max-routes 12 \
  --limit-train 128 \
  --limit-val 64 \
  --device cuda \
  --output-dir /kaggle/working/runs \
  --run-name smoke_test
```

This confirms:

- dataset loading works
- training loop runs
- validation works
- checkpoints are written to `/kaggle/working/runs/smoke_test`

## 6. Full-ish Validation Run

```bash
!python scripts/train_visdrone.py \
  --visdrone-yaml /kaggle/input/visdrone-dataset/VisDrone.yaml \
  --dataset-root /kaggle/input/visdrone-dataset \
  --epochs 5 \
  --batch-size 8 \
  --image-size 640 \
  --patch-size 96 \
  --max-routes 12 \
  --device cuda \
  --output-dir /kaggle/working/runs \
  --run-name visdrone_h2r
```

Then evaluate:

```bash
!python scripts/evaluate_visdrone.py \
  --checkpoint /kaggle/working/runs/visdrone_h2r/best.pt \
  --visdrone-yaml /kaggle/input/visdrone-dataset/VisDrone.yaml \
  --dataset-root /kaggle/input/visdrone-dataset \
  --split val \
  --batch-size 8 \
  --device cuda \
  --save-json /kaggle/working/visdrone_eval.json
```

## 7. Download Artifacts

After the notebook finishes, download from `/kaggle/working/`:

- `runs/visdrone_h2r/best.pt`
- `runs/visdrone_h2r/last.pt`
- `runs/visdrone_h2r/history.json`
- `visdrone_eval.json`

## Practical Notes

- Start with `--image-size 512` if GPU memory is tight.
- Reduce `--batch-size` before reducing `--max-routes`.
- Use `--limit-train` and `--limit-val` first to verify paths and labels.
- The repo currently reports `AP50 proxy`, not the official VisDrone evaluation server metric.
