# LBM Eraser

**Goal**: reproduce the object-removal result from the LBM paper (Section 4.1) using SD1.5.
Non-official implementation.

## Install

One option is to install [uv](https://docs.astral.sh/uv/getting-started/installation/) and then:

```bash
uv venv -p 3.10
uv pip sync requirements.txt
uv pip install -e .
```

## Prepare RORD Dataset

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

### Export a "lite" validation set

To improve the diversity of the validation set, we create a `val-lite` version of it, by choosing only one frame per scene in RORD validation set.

```
uv run --script contrib/eraser/datasets/extract_val_lite.py
```

## Prepare Re-LAION Dataset

We extract 1.3M images from [Re-LAION-Caption19M](https://huggingface.co/datasets/supermodelresearch/Re-LAION-Caption19M) for inpainting (see paper 4.1: "in-the-wild images where we randomly masked an area of the image")

### Requirements

1/ Install [duckdb](https://duckdb.org/) CLI
```bash
curl https://install.duckdb.org | sh
```

2/ Install [img2dataset](https://github.com/rom1504/img2dataset)
```bash
uv tool install -p 3.10 img2dataset
```

3/ Set a knot resolver by following [img2dataset instructions](https://github.com/rom1504/img2dataset?tab=readme-ov-file#setting-up-a-knot-resolver)

### Download

First download the files
```bash
git lfs install
git clone git@hf.co:datasets/supermodelresearch/Re-LAION-Caption19M data/supermodelresearch/Re-LAION-Caption19M
```
Then extract the images with `aestetic > 0.5` and `pwatermark < 0.2` (takes ~50s)
```bash
duckdb < contrib/eraser/datasets/re_laion_caption19m_filter.sql
```
An extract of `Re-LAION-Caption19M` is then saved in `data/Re-LAION-1300K-parquet`

It takes ~4h (on a `c6i.32xlarge`) and it requires ~1.1TB of free space disk to convert into WebDataset format
```bash
export DATA_FOLDER='data/Re-LAION-1300K-parquet/'
export OUTPUT_FOLDER=data/Re-LAION-1300K
export SAVE_ADDITIONAL_COLUMNS='["aesthetic_score","llava_next_caption","luminance_score","ocr_score","llava_next_caption_shuffled","similarity","punsafe","pwatermark"]'
export PROCESSES_COUNT=128
export THREAD_COUNT=256

NO_ALBUMENTATIONS_UPDATE=1 uv run img2dataset \
  --url_list $DATA_FOLDER \
  --output_folder $OUTPUT_FOLDER \
  --input_format parquet \
  --output_format webdataset \
  --processes_count $PROCESSES_COUNT\
  --thread_count $THREAD_COUNT\
  --min_image_size 256 \
  --number_sample_per_shard 1024 \
  --resize_mode no \
  --max_shard_retry 5 \
  --max_image_area 16777216 \
  --save_additional_columns $SAVE_ADDITIONAL_COLUMNS
```

Notes : 
* We purposely do not resize the images here so the dataset is compatible with multiple sizes
* Setting `--resize_mode no` is mandatory, by default it's `--resize_mode border` with `--image_size 256`
* Setting `--max_image_area 16777216 = 4096x4096` is made to avoid errors like `PIL.Image.DecompressionBombError`
* Currently we only set `--min_image_size 256` as a sanity check, but the dataset is supposed to contain only 1024 images already
* If the script is hanging, just kill it and re-run it, it'll resume smoothly

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
uv run python contrib/eraser/training/train_eraser.py contrib/eraser/training/config/eraser.yaml
```

