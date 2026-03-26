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

If your dataset does not contain `VisDrone.yaml`, you can point `--visdrone-yaml` at the dataset directory itself and still pass `--dataset-root`.

Examples:

```bash
--visdrone-yaml /kaggle/input/visdrone-dataset/VisDrone.yaml
```

or, if no YAML file exists:

```bash
--visdrone-yaml /kaggle/input/visdrone-dataset
```

If you are unsure about the exact Kaggle mount path, inspect it first:

```bash
!ls /kaggle/input
!find /kaggle/input -maxdepth 3 | head -200
```

## 5. Confirm Both GPUs Are Visible

Run:

```bash
!python -c "import torch; print('cuda', torch.cuda.is_available()); print('num_gpus', torch.cuda.device_count())"
```

On your setup, you want to see `num_gpus 2`.

## 6. Fast Smoke Test On A Small Subset

For a real 2-GPU smoke test, use `torchrun` and treat `--batch-size` as per-GPU batch size:

```bash
!torchrun --standalone --nproc_per_node=2 scripts/train_visdrone.py \
  --visdrone-yaml /kaggle/input/visdrone-dataset \
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
- distributed training on both GPUs runs
- validation works
- checkpoints are written to `/kaggle/working/runs/smoke_test`

If this goes out of memory, reduce `--batch-size` from `4` to `2` before touching the model settings.

## 7. Full Pipeline Run On 2x T4

Recommended starting point for 2x T4:

- `--batch-size 4` per GPU
- global batch size = `8`
- `--image-size 640`
- `--patch-size 96`
- `--max-routes 12`
- `--amp` enabled by default

Train:

```bash
!torchrun --standalone --nproc_per_node=2 scripts/train_visdrone.py \
  --visdrone-yaml /kaggle/input/visdrone-dataset \
  --dataset-root /kaggle/input/visdrone-dataset \
  --epochs 20 \
  --batch-size 4 \
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
  --visdrone-yaml /kaggle/input/visdrone-dataset \
  --dataset-root /kaggle/input/visdrone-dataset \
  --split val \
  --batch-size 8 \
  --device cuda \
  --save-json /kaggle/working/visdrone_eval.json
```

## 8. Download Artifacts

After the notebook finishes, download from `/kaggle/working/`:

- `runs/visdrone_h2r/best.pt`
- `runs/visdrone_h2r/last.pt`
- `runs/visdrone_h2r/history.json`
- `visdrone_eval.json`

## Practical Notes

- On 2x T4, start with `--batch-size 4` per GPU at `640`.
- If you hit OOM, move to `--batch-size 2` per GPU before reducing `--image-size`.
- Start with `--image-size 512` only if `640` is still too heavy.
- Reduce `--batch-size` before reducing `--max-routes`.
- Use `--limit-train` and `--limit-val` first to verify paths and labels.
- The repo currently reports `AP50 proxy`, not the official VisDrone evaluation server metric.
- `torchrun` examples here are for Kaggle Linux. Local Windows testing may need different rendezvous settings.
