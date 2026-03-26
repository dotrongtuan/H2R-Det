# H2R-Det

`H2R-Det` is a lightweight research codebase for validating a sparse, human-aware routing strategy for tiny-person detection:

- a `global scout` processes the full image
- a `human-aware routing field` proposes a small set of high-value regions
- a `sparse zoom expert` refines only those regions
- training tracks both `human-focused accuracy proxies` and `compute budget proxies`

The current repository is designed to be:

- easy to push to GitHub
- easy to smoke-test on Kaggle
- honest about scope: it is a research prototype, not yet an official VisDrone challenge submission

## What Is Already Implemented

- end-to-end PyTorch model for sparse routing and human refinement
- synthetic sanity benchmark for feasibility testing
- VisDrone YAML parser and dataloader scaffold
- training loop with checkpoints and validation
- evaluation script with `AP50 proxy`, `human AP50`, routing recall, and routed-area metrics

## Important Scope Note

The validation script currently reports:

- `AP50 proxy` from the repo's own decoding and matching pipeline
- `human AP50` for the `pedestrian` and `people/person` classes
- `routing recall`
- `mean routed area fraction`

This is useful for research iteration, but it is **not** the same as the official VisDrone challenge metric pipeline.

## Installation

```bash
python -m pip install -r requirements.txt
```

Or as a package:

```bash
python -m pip install -e .
```

## Quick Feasibility Check

```bash
python scripts/sanity_check.py --steps 80 --batch-size 2 --image-size 256 --patch-size 64 --max-routes 12
```

On the current prototype, this sanity run demonstrated that routing can learn to recover tiny-person regions while the sparse refine branch uses much less compute than a dense sliding alternative.

## Train On VisDrone

```bash
python scripts/train_visdrone.py ^
  --visdrone-yaml path\\to\\VisDrone.yaml ^
  --dataset-root path\\to\\visdrone_root ^
  --epochs 5 ^
  --batch-size 4 ^
  --image-size 640 ^
  --patch-size 96 ^
  --max-routes 12 ^
  --device cuda
```

Useful options:

- `--limit-train 200` for a fast smoke test
- `--limit-val 100` for a quick validation pass
- `--output-dir runs` to control checkpoint output
- `--batch-size` is per process; under multi-GPU the global batch is `batch_size * world_size`

### Multi-GPU

The training script supports `torchrun` distributed training. Example on 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_visdrone.py ^
  --visdrone-yaml path\\to\\VisDrone.yaml ^
  --dataset-root path\\to\\visdrone_root ^
  --epochs 5 ^
  --batch-size 4 ^
  --image-size 640 ^
  --patch-size 96 ^
  --max-routes 12 ^
  --device cuda ^
  --output-dir runs ^
  --run-name visdrone_h2r_ddp
```

Artifacts are written to `runs/<run_name>/`:

- `best.pt`
- `last.pt`
- `config.json`
- `history.json`

## Evaluate A Checkpoint

```bash
python scripts/evaluate_visdrone.py ^
  --checkpoint runs\\my_run\\best.pt ^
  --visdrone-yaml path\\to\\VisDrone.yaml ^
  --dataset-root path\\to\\visdrone_root ^
  --split val ^
  --batch-size 4 ^
  --device cuda
```

## Repository Layout

- `src/h2r_det/model.py`: model and sparse routing logic
- `src/h2r_det/losses.py`: training losses
- `src/h2r_det/metrics.py`: routing and AP50 proxy metrics
- `src/h2r_det/visdrone.py`: YAML parsing, resizing, dataloaders
- `scripts/sanity_check.py`: synthetic feasibility test
- `scripts/train_visdrone.py`: real-data training entrypoint
- `scripts/evaluate_visdrone.py`: checkpoint evaluation entrypoint
- `docs/KAGGLE.md`: Kaggle deployment guide

## Kaggle

The practical Kaggle workflow is documented here:

[docs/KAGGLE.md](docs/KAGGLE.md)
