from typing import Callable, List, Union, Optional

import pytorch_lightning as pl
from sympy import N
import webdataset as wds
from lbm.data.datasets.dataset import DataPipeline
from lbm.data.filters import BaseFilter, FilterWrapper
from lbm.data.mappers import BaseMapper, MapperWrapper
from lbm.data.datasets.collation_fn import custom_collation_fn
from lbm.data.datasets.datasets_config import DataModuleConfig
from lbm.config import BaseConfig
import torch
FilterMapper = Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
FilterMappers = List[FilterMapper]
from dataclasses import dataclass
from torch.utils.data import IterableDataset, DataLoader


@dataclass
class DataPipelineConfig(BaseConfig):
    """
    Configuration for a single DataPipeline

    Args:

        data_module_config (DataModuleConfig):
            Configuration for the training dataset

        filters_mappers (FilterMappers):
            List of filters and mappers for the training dataset. These will be sequentially applied.

        batched_filters_mappers (Optional[FilterMappers]):
            List of batched transforms for the training dataset. These will be sequentially applied.
        
        batched_fn (Callable):
            Function to use for batching the dataset. Defaults to wds.batched.
    """
    
    data_module_config: DataModuleConfig
    name: str
    filters_mappers: FilterMappers
    batched_filters_mappers: Optional[FilterMappers] = None
    batched_fn: Callable = wds.batched

class MixedIterableDataset(IterableDataset):
    def __init__(self, iterable_datasets: List[IterableDataset], weights: Optional[List[float]] = None, seed: Optional[int] = None):
        """
        Mix multiple IterableDatasets according to given weights.

        The loop stops when the shortest IterableDataset is exhausted.

        Args:
            iterable_datasets: list of iterable_datasets
            weights: list of floats, unnormalized
            generator: torch.Generator for deterministic randomness
        """
        super().__init__()
        self.iterable_datasets = iterable_datasets
        n = len(iterable_datasets)
        if weights is None:
            weights = [1.0 / n] * n
        total = sum(weights)
        self.weights = [w / total for w in weights]
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = None

    def __iter__(self):
        # Important: each IterableDataset is already iterable
        iterators = [iter(ds) for ds in self.iterable_datasets]
        while True:
            idx = torch.multinomial(torch.tensor(self.weights), 1, generator=self.generator).item()
            try:
                yield next(iterators[idx])
            except StopIteration:
                break


class HybridDataModule(pl.LightningDataModule):
    """
    Main DataModule class for creating data loaders and training/evaluating models

    Args:

        train_pipeline_configs (List[DataPipelineConfig]):
            List of DataPipelineConfig for each dataset pipeline
        
        train_weights (Optional[List[float]]):
            List of weights for each training dataset pipeline. Defaults to None.
        
        train_seed (Optional[int]):
            Random seed for training dataloader sampling. Defaults to None.

        eval_pipeline_configs (Optional[List[DataPipelineConfig]]):
            List of DataPipelineConfig for each dataset pipeline. Defaults to None.
        
        eval_weights (Optional[List[float]]):
            List of weights for each evaluation dataset pipeline. Defaults to None.
        
        eval_seed (Optional[int]):
            Random seed for evaluation dataloader sampling. Defaults to None.
    """

    def __init__(
        self,
        train_pipeline_configs: List[DataPipelineConfig],
        eval_pipeline_configs: Optional[List[DataPipelineConfig]] = None,
        train_weights: Optional[List[float]] = None,
        train_seed: Optional[int] = None,
        eval_weights: Optional[List[float]] = None,
        eval_seed: Optional[int] = None,
        train_num_workers: int = 10,
        eval_num_workers: int = 10,
    ):
        super().__init__()
        self.train_pipeline_configs = train_pipeline_configs
        self.eval_pipeline_configs = eval_pipeline_configs
        self.train_weights = train_weights
        self.train_seed = train_seed
        self.eval_weights = eval_weights
        self.eval_seed = eval_seed
        self.train_num_workers = train_num_workers
        self.eval_num_workers = eval_num_workers

    def setup(self, stage=None):
        """
        Setup the data module and create the webdataset processing pipelines
        """


        self.train_data_pipelines : List[DataPipeline] = [
            DataPipeline(
                config.data_module_config,
                config.filters_mappers,
                config.batched_filters_mappers,
                batched_fn=config.batched_fn,
            )
            for config in self.train_pipeline_configs
        ]

        for pipeline in self.train_data_pipelines:
            pipeline.setup()

        if self.eval_pipeline_configs is not None:
            self.eval_data_pipelines : List[DataPipeline] = [
                DataPipeline(
                    config.data_module_config,
                    config.filters_mappers,
                    config.batched_filters_mappers,
                    batched_fn=config.batched_fn,
                )
                for config in self.eval_pipeline_configs
            ]

            for pipeline in self.eval_data_pipelines:
                pipeline.setup()


    def _build_mixed_dataloader(self, data_pipelines: List[DataPipeline], num_workers: int, weights: Optional[List[float]] = None, seed: Optional[int] = None) -> DataLoader:
        iterable_datasets = [
            pipeline.pipeline for pipeline in data_pipelines
        ]
        mixed_dataset = MixedIterableDataset(
            iterable_datasets,
            weights=weights,
            seed=seed,
        )
        return DataLoader(
            mixed_dataset,
            batch_size=None,
            num_workers=num_workers,
        )
        
    def train_dataloader(self) -> DataLoader:
        return self._build_mixed_dataloader(
            self.train_data_pipelines,
            num_workers=self.train_num_workers,
            weights=self.train_weights,
            seed=self.train_seed,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.eval_pipeline_configs is not None

        return self._build_mixed_dataloader(
            self.eval_data_pipelines,
            num_workers=self.eval_num_workers,
            weights=self.eval_weights,
            seed=self.eval_seed,
        )
