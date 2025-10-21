from curses import meta
from typing import Any
import pytorch_lightning as pl
import torch
from lbm import data
from lbm.contrib.data.inpainter_dataset import InpainterDataset
import logging
from lbm.contrib.data.sana import aspect_ratios
from lbm.contrib.data.sana.wids import DistributedRangedSampler
from lbm.contrib.data.sana.data_sampler import AspectRatioBatchSampler
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
logger = logging.getLogger(__name__)
import time
import torch.distributed as dist
import json
from pathlib import Path
@dataclass
class InpainterDataModuleConfig:
    resolution: int
    num_workers: int
    batch_size: int
    data_dir: str | list[str]
    meta_path: None | str | list[str] = None # if none, will try to find wids-meta.json in data_dir

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1

def is_global_zero():
    return get_rank() == 0 or not is_distributed() or get_rank() == -1

def barrier():
    if is_distributed():
        dist.barrier()

def rank_zero_info(msg):
    if is_global_zero():
        logger.info(msg)


def collate_fn(batch: dict[str, Any]) -> dict[str, Any]:
    img_hw = [data["img_hw"] for data in batch]
    indexes = [data["index"] for data in batch]
    
    # Check that all 'before' and 'after' tensors have the same shape
    before_shapes = [data["before"].shape for data in batch]
    if len(set(before_shapes)) != 1:
        print(batch[0].keys())
        raise ValueError(f"Inconsistent 'before' shapes in batch: {img_hw}, try to reset your cache files.")

    return {
        "img_hw": [data["img_hw"] for data in batch],
        "aspect_ratio": [data["aspect_ratio"] for data in batch],
        "index": [data["index"] for data in batch],
        "shard": [data["shard"] for data in batch],
        "shardindex": [data["shardindex"] for data in batch],
        "before": torch.stack([data["before"] for data in batch], dim=0),
        "after": torch.stack([data["after"] for data in batch], dim=0),
        "uid": [data["uid"] for data in batch],
    }

def cache_dataset_for_batch_sampler(dataloader: DataLoader, cache_file: str):
    """
    Caches dataset indices for multi-scale training

    Inpired from https://github.com/NVlabs/Sana/blob/34b4aa5c102bfa36cb57ad68a43a15ec7fe3f411/train_scripts/train.py#L920
    """
    print('in ')

    rank_zero_info(
        f"Start caching dataset for batch_sampler at {cache_file}. "
        "This may take a while... Training will not start until caching completes."
    )

    # Initialize sampler offset if resuming
    sampler = dataloader.batch_sampler.sampler
    sampler.set_start(max(getattr(dataloader.batch_sampler, "exist_ids", 0), 0))

    barrier()

    for index, _ in enumerate(dataloader):
        barrier()

        if index % 2000 == 0 and is_global_zero():
            current_len = len(dataloader.batch_sampler.cached_idx)
            total_len = len(dataloader)
            rank_zero_info(f"Cached {current_len} / {total_len}")

        # Save when nearly complete
        if len(dataloader.batch_sampler.cached_idx) >= len(dataloader) - 1000:
            if is_global_zero():
                current_len = len(dataloader.batch_sampler.cached_idx)
                total_len = len(dataloader)
                rank_zero_info(f"Saving cache ({current_len}/{total_len}) to {cache_file}")
                json.dump(dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
        barrier()

    barrier()

    if is_global_zero():
        final_len = len(dataloader.batch_sampler.cached_idx)
        rank_zero_info(f"Final cached length: {final_len}")
        json.dump(dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)

class InpainterDataModule(pl.LightningDataModule):
    """
    Wrapper between PyTorch Lightning and InpainterDataset multi-scale data loader.
    """

    def __init__(
        self,
        train_config: InpainterDataModuleConfig,
        val_config: InpainterDataModuleConfig,
    ):
        super().__init__()

        self.train_config = train_config
        self.val_config = val_config
        self.cache_dir = Path(f"~/.cache/_wids_batchsampler_cache").expanduser()
    
    def get_cache_file(self, cache_dir: Path, num_replicas: int, rank: int, config: InpainterDataModuleConfig) -> Path:

        cache_dir_str = "-".join([i.replace("/", "--") for i in config.data_dir])  # to avoid slashes
        if config.meta_path is not None:
            meta_path_str = "-".join(
                [i.replace("/", "--").replace(".", "--") for i in config.meta_path if i is not None]
            )
            cache_dir_str += f"-{meta_path_str}"
        cache_file = cache_dir / f"num_replicas{num_replicas}-rank{rank}-{cache_dir_str}.json"
        return cache_file

    def setup(self, stage: None | str = None) -> None:
        """
        Setup the data module and create the webdataset processing pipelines
        """
        if not dist.is_initialized():
            logger.info("Distributed not initialized, setting num_replicas=1 and rank=0")
            num_replicas = 0
            rank = 1
        else:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

        self.train_dataset = InpainterDataset(
            data_dir=self.train_config.data_dir,
            resolution=self.train_config.resolution,
            meta_path=self.train_config.meta_path,
            num_replicas=num_replicas,
            cache_dir=self.cache_dir,
        )
        self.val_dataset = InpainterDataset(
            data_dir=self.val_config.data_dir,
            resolution=self.val_config.resolution,
            meta_path=self.val_config.meta_path,
            num_replicas=num_replicas,
            cache_dir=self.cache_dir,
        )
        self.train_batch_sampler = self._prepare_sampler(
            num_replicas, rank, self.train_config, self.train_dataset
        )
        self.val_batch_sampler = self._prepare_sampler(
            num_replicas, rank, self.val_config, self.val_dataset
        )
        
    def _prepare_sampler(self, num_replicas: int, rank: int, config: InpainterDataModuleConfig, dataset: Dataset) -> AspectRatioBatchSampler:

        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True) 

        cache_file = self.get_cache_file(
            self.cache_dir, num_replicas, rank, self.train_config
        )

        sampler = DistributedRangedSampler(self.val_dataset, num_replicas=num_replicas, rank=rank)
        batch_sampler = AspectRatioBatchSampler(
            sampler=sampler,
            dataset=self.train_dataset,
            batch_size=self.train_config.batch_size,
            aspect_ratios=self.train_dataset.get_aspect_ratios(),
            drop_last=True,
            cache_file=str(cache_file),
            caching=True,
            ratio_nums=self.train_dataset.get_ratio_nums(),
        )

        dataloader = self._build_dataloader(batch_sampler, dataset, config.num_workers)
        cache_dataset_for_batch_sampler(dataloader, cache_file)

    def _build_dataloader(self, batch_sampler: AspectRatioBatchSampler, dataset: Dataset, num_workers: int) -> DataLoader:
        """
        Inspired from https://github.com/NVlabs/Sana/blob/48ec430646bc524bbf0bb5d68bf092739b0f5082/train_scripts/train.py#L907-L908
        """
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


    def train_dataloader(self):
        return self._build_dataloader(self.train_batch_sampler, self.train_dataset, self.train_config.num_workers)

    def val_dataloader(self):
        return self._build_dataloader(self.val_batch_sampler, self.val_dataset, self.val_config.num_workers)

if __name__ == "__main__":

    image_size = 256
    data_dir = ["/home/piercus/webdatasets/Re-LAION-1300K/"]
    meta_path = ["/home/piercus/webdatasets/Re-LAION-1300K/wids-meta-train.json"]
    data_module = InpainterDataModule(
        train_config=InpainterDataModuleConfig(
            resolution=image_size,
            num_workers=4,
            batch_size=2,
            data_dir=data_dir,
            meta_path=meta_path
        ),
        val_config=InpainterDataModuleConfig(
            resolution=image_size,
            num_workers=4,
            batch_size=2,
            data_dir=data_dir,
            meta_path=meta_path
        )
    )
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    max_num = 200
    num = 0
    for data in train_dataloader:
        # Test something here
        print(data["before"].shape)
        if num >= max_num:
            break
        num += 1
