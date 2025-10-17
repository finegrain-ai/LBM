from curses import meta
import pytorch_lightning as pl
from lbm.contrib.data.inpainter_dataset import InpainterDataset
import logging
from lbm.contrib.data.sana import aspect_ratios
from lbm.contrib.data.sana.wids import DistributedRangedSampler
from lbm.contrib.data.sana.data_sampler import AspectRatioBatchSampler
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
logger = logging.getLogger(__name__)

@dataclass
class InpainterDataModuleConfig:
    resolution: int
    num_workers: int
    batch_size: int
    data_dir: str | list[str]
    meta_path: None | str | list[str] = None # if none, will try to find wids-meta.json in data_dir

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

    def setup(self, stage=None):
        """
        Setup the data module and create the webdataset processing pipelines
        """

        self.train_dataset = InpainterDataset(
            data_dir=self.train_config.data_dir,
            resolution=self.train_config.resolution,
            meta_path=self.train_config.meta_path
        )
        self.val_dataset = InpainterDataset(
            data_dir=self.val_config.data_dir,
            resolution=self.val_config.resolution,
            meta_path=self.val_config.meta_path
        )

    def _build_dataloader(self, config: InpainterDataModuleConfig, dataset: Dataset) -> DataLoader:
        """
        Inspired from https://github.com/NVlabs/Sana/blob/48ec430646bc524bbf0bb5d68bf092739b0f5082/train_scripts/train.py#L907-L908
        """
        sampler = DistributedRangedSampler(dataset)

        batch_sampler = AspectRatioBatchSampler(
            sampler=sampler,
            dataset=dataset,
            batch_size=config.batch_size,
            aspect_ratios=dataset.get_aspect_ratios(),
            drop_last=True,
            ratio_nums=dataset.get_ratio_nums(),
        )
        return DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=config.num_workers, pin_memory=True
        )


    def train_dataloader(self):
        return self._build_dataloader(self.train_config, self.train_dataset)

    def val_dataloader(self):
        return self._build_dataloader(self.val_config, self.val_dataset)