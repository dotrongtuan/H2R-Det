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

You now have three options:

1. pass a real YAML file
2. pass the dataset directory
3. pass the built-in alias `VisDrone.yaml`

Examples:

```bash
--visdrone-yaml /kaggle/input/visdrone-dataset/VisDrone.yaml
```

or, if no YAML file exists:

```bash
--visdrone-yaml /kaggle/input/visdrone-dataset
```

or:

```bash
--visdrone-yaml VisDrone.yaml
```

When you use the built-in alias `VisDrone.yaml`, the code will:

- look for an already-mounted VisDrone dataset under `/kaggle/input`
- if found, use it directly
- otherwise download and prepare the dataset into a writable cache directory

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
  --visdrone-yaml VisDrone.yaml \
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

- `--batch-size 2` per GPU
- global batch size = `4`
- `--image-size 640`
- `--patch-size 96`
- `--max-routes 24`
- `--amp` enabled by default
- `--early-stop-patience 4`
- `--early-stop-min-delta 0.0005`
- `--min-epochs 10`

Train:

```bash
!torchrun --standalone --nproc_per_node=2 scripts/train_visdrone.py \
  --visdrone-yaml VisDrone.yaml \
  --epochs 20 \
  --batch-size 2 \
  --image-size 640 \
  --patch-size 96 \
  --max-routes 24 \
  --early-stop-patience 4 \
  --early-stop-min-delta 0.0005 \
  --min-epochs 10 \
  --device cuda \
  --output-dir /kaggle/working/runs \
  --run-name visdrone_h2r
```

Then evaluate:

```bash
!python scripts/evaluate_visdrone.py \
  --checkpoint /kaggle/working/runs/visdrone_h2r/best.pt \
  --visdrone-yaml VisDrone.yaml \
  --split val \
  --batch-size 2 \
  --device cuda \
  --save-json /kaggle/working/visdrone_eval.json
```

## 8. One-Cell Full Pipeline With Fallback

If you want a single Kaggle cell that:

- clone or pull the repo
- install the package
- log `nvidia-smi`
- train with AMP first
- automatically retry with `--no-amp` if the AMP run fails
- evaluate the best checkpoint
- pack the artifacts into one tarball

use:

```bash
%%bash
set -euo pipefail

REPO_DIR="/kaggle/working/h2r-det"
RUN_NAME="visdrone_h2r_full"
RUN_DIR="/kaggle/working/runs/${RUN_NAME}"
EVAL_JSON="/kaggle/working/${RUN_NAME}_eval.json"
ARTIFACT_TAR="/kaggle/working/${RUN_NAME}_artifacts.tar.gz"
SMI_LOG="/kaggle/working/${RUN_NAME}_nvidia_smi.txt"

cd /kaggle/working

if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/dotrongtuan/H2R-Det.git h2r-det
else
  cd "$REPO_DIR"
  git pull
fi

cd "$REPO_DIR"
python -m pip install -e . --no-deps

python - <<'PY'
import torch
print("torch =", torch.__version__)
print("cuda =", torch.cuda.is_available())
print("gpu_count =", torch.cuda.device_count())
PY

nvidia-smi | tee "$SMI_LOG"

mkdir -p /kaggle/working/runs

set +e
torchrun --standalone --nproc_per_node=2 scripts/train_visdrone.py \
  --visdrone-yaml VisDrone.yaml \
  --epochs 20 \
  --batch-size 2 \
  --image-size 640 \
  --patch-size 96 \
  --max-routes 24 \
  --early-stop-patience 4 \
  --early-stop-min-delta 0.0005 \
  --min-epochs 10 \
  --device cuda \
  --log-interval 100 \
  --output-dir /kaggle/working/runs \
  --run-name "${RUN_NAME}"
TRAIN_STATUS=$?
set -e

if [ "$TRAIN_STATUS" -ne 0 ]; then
  echo "AMP run failed, retrying with --no-amp"
  rm -rf "$RUN_DIR"
  torchrun --standalone --nproc_per_node=2 scripts/train_visdrone.py \
    --visdrone-yaml VisDrone.yaml \
    --epochs 20 \
    --batch-size 2 \
    --image-size 640 \
    --patch-size 96 \
    --max-routes 24 \
    --early-stop-patience 4 \
    --early-stop-min-delta 0.0005 \
    --min-epochs 10 \
    --device cuda \
    --no-amp \
    --log-interval 100 \
    --output-dir /kaggle/working/runs \
    --run-name "${RUN_NAME}"
fi

python scripts/evaluate_visdrone.py \
  --checkpoint "${RUN_DIR}/best.pt" \
  --visdrone-yaml VisDrone.yaml \
  --split val \
  --batch-size 2 \
  --device cuda \
  --save-json "${EVAL_JSON}"

tar -czf "${ARTIFACT_TAR}" \
  -C /kaggle/working \
  "runs/${RUN_NAME}" \
  "$(basename "${EVAL_JSON}")" \
  "$(basename "${SMI_LOG}")"

echo
echo "Done."
echo "Run dir: ${RUN_DIR}"
echo "Eval json: ${EVAL_JSON}"
echo "Artifact tar: ${ARTIFACT_TAR}"
ls -lah "${RUN_DIR}"
cat "${EVAL_JSON}"
```

Notes:

- this cell assumes the Kaggle notebook has access to `git clone`
- if VisDrone is not mounted under `/kaggle/input`, built-in alias mode may need Internet access to auto-download the dataset
- if the AMP run fails for any reason, the cell retries from scratch with `--no-amp`

## 9. Download Artifacts

After the notebook finishes, download from `/kaggle/working/`:

- `runs/<run_name>/best.pt`
- `runs/<run_name>/last.pt`
- `runs/<run_name>/history.json`
- `<run_name>_eval.json`
- `<run_name>_artifacts.tar.gz`

## Practical Notes

- On 2x T4, start with `--batch-size 2` per GPU at `640`.
- If you hit OOM, move to `--batch-size 2` per GPU before reducing `--image-size`.
- Start with `--image-size 512` only if `640` is still too heavy.
- Reduce `--batch-size` before reducing `--max-routes`.
- Use `--limit-train` and `--limit-val` first to verify paths and labels.
- If the VisDrone Kaggle dataset is already mounted, `--visdrone-yaml VisDrone.yaml` should now find it automatically.
- The repo currently reports `AP50 proxy`, not the official VisDrone evaluation server metric.
- `torchrun` examples here are for Kaggle Linux. Local Windows testing may need different rendezvous settings.
