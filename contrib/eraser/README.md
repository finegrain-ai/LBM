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

## Train

To train the model on a single GPU machine, you can use the following command.
Works on a single RTX 3090:

```bash
# Mock the SLURM env variables
export SLURM_JOB_ID=0000 
export SLURM_PROCID=0 
export SLURM_ARRAY_TASK_ID=0
export SLURM_NPROCS=1
export SLURM_NODEID=0
export SLURM_NNODES=1
export SLURM_LOCALID=0 

# Neptune (https://neptune.ai) settings. Use 'offline' mode to disable logging.
export NEPTUNE_MODE=async
export NEPTUNE_API_TOKEN="YOUR_API_TOKEN"
python contrib/eraser/training/train_eraser.py contrib/eraser/training/config/eraser.yaml
```
