## LBM Eraser

**Goal**: reproduce the object-removal result from the LBM paper (Section 4.1) using SD1.5.
Non-official implementation.

### Dataset

Prepare a group of [webdataset](https://github.com/webdataset/webdataset) shards, organized like:

```
<id0>.after.jpg
<id0>.before.jpg
<id0>.mask.png
<id1>.after.jpg
<id1>.before.jpg
<id1>.mask.png
...
```

### Train

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

# dataset
export PATH_TO_TRAIN_TARS=...
export PATH_TO_VAL_TARS=...

python examples/training/train_eraser.py examples/training/config/eraser.yaml
```