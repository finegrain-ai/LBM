# LBM Eraser

**Goal**: reproduce the object-removal result from the LBM paper (Section 4.1) using SD1.5.
Non-official implementation.

## Dataset

### Download the RORD dataset locally

```
uv tool install 'huggingface_hub[cli]'
sudo apt-get install p7zip-full
./contrib/eraser/datasets/download_rord.sh
```

### Format the data to webdataset

ETA on AWS `c6i.8xlarge`: 2h30
```
uv run --script contrib/eraser/datasets/preprocess_rord.py
```

Result is saved in `data/RORD-processed/` (auto-created if not existing)

### Val-lite

To improve the diversity of the validation set, we create a `val-lite` version of it, by choosing only one frame per scene in RORD validation set.

```
python contrib/eraser/datasets/extract_val_lite.py
```

## Train

```bash
# Neptune (https://neptune.ai) settings. Use 'offline' mode to disable logging.
export NEPTUNE_MODE=async
export NEPTUNE_API_TOKEN="YOUR_API_TOKEN"
```

To limit to a single GPU
```bash
export CUDA_VISIBLE_DEVICES=0
```

Then
```bash
python contrib/eraser/training/train_eraser.py contrib/eraser/training/config/eraser.yaml
```

