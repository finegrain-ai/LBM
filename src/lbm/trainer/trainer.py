import importlib
import logging
import re
import time
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import piq

from ..models.base.base_model import BaseModel
from .training_config import TrainingConfig
from .utils import mix_images_column_wise

logging.basicConfig(level=logging.INFO)

class MetricHandler:
    def __init__(self, metrics: list[str], device: torch.device) -> None:
        self.metrics = metrics
        self.lpips_metric = piq.LPIPS() if "lpips" in metrics else None
        self.pieapp_metric = piq.PieAPP() if "pieapp" in metrics else None
        self.dists_metric = piq.DISTS() if "dists" in metrics else None
        self.to(device)

    def compute(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
        """
        Compute the metrics between the prediction and the target
        It returns an average over the batch
        
        Args:
            prediction (torch.Tensor): The predicted image tensor of shape (B, C, H, W), normalized in [-1, 1]
            target (torch.Tensor): The target image tensor of shape (B, C, H, W), normalized in [-1, 1]
        Returns:
            dict[str, Any]: A dictionary containing the computed metrics
        """

        # From [-1, 1] to [0, 1] for PIQ metrics
        prediction = ((prediction + 1) / 2.0).clamp(0,1).to(self.device)
        target = ((target + 1) / 2.0).clamp(0,1).to(self.device)

        results = {}
        for metric in self.metrics:
            match metric:
                case "lpips":
                    assert self.lpips_metric is not None
                    results["lpips"] = self.lpips_metric(prediction, target)
                case "pieapp":
                    assert self.pieapp_metric is not None
                    results["pieapp"] = self.pieapp_metric(prediction, target)
                case "dists":
                    assert self.dists_metric is not None
                    results["dists"] = self.dists_metric(prediction, target)
                case "psnr":
                    results["psnr"] = piq.psnr(prediction, target)
                case _:
                    raise ValueError(f"Unknown metric {metric}")
        return results

    def to(self, device: torch.device) -> None:
        self.lpips_metric.to(device)
        self.pieapp_metric.to(device)
        self.dists_metric.to(device)
        self.device = device

class TrainingPipeline(pl.LightningModule):
    """
    Main Training Pipeline class

    Args:

        model (BaseModel): The model to train
        pipeline_config (TrainingConfig): The configuration for the training pipeline
        verbose (bool): Whether to print logs in the console. Default is False.
    """

    def __init__(
        self,
        model: BaseModel,
        pipeline_config: TrainingConfig,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.pipeline_config = pipeline_config
        self.log_samples_model_kwargs = pipeline_config.log_samples_model_kwargs

        # save hyperparameters.
        self.save_hyperparameters(ignore="model")
        self.save_hyperparameters({"model_config": model.config.to_dict()})

        # logger.
        self.verbose = verbose

        # setup logging.
        log_keys = pipeline_config.log_keys

        if isinstance(log_keys, str):
            log_keys = [log_keys]

        if log_keys is None:
            log_keys = []

        self.log_keys = log_keys
        self.column_visualization_keys = pipeline_config.column_visualization_keys
        self.metric_handler = MetricHandler(pipeline_config.metrics, device=self.device) if pipeline_config.metrics is not None else None

    def on_fit_start(self) -> None:
        self.model.on_fit_start(device=self.device)
        if self.global_rank == 0:
            self.timer = time.perf_counter()

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        if self.global_rank == 0:
            logging.debug("on_train_batch_end")
        self.model.on_train_batch_end(batch)

        average_time_frequency = 10
        if self.global_rank == 0 and batch_idx % average_time_frequency == 0:
            delta = time.perf_counter() - self.timer
            logging.info(
                f"Average time per batch {batch_idx} took {delta / (batch_idx + 1)} seconds"
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Setup optimizers and learning rate schedulers.
        """
        optimizers = []
        lr = self.pipeline_config.learning_rate
        param_list = []
        n_params = 0
        param_list_ = {"params": []}
        for name, param in self.model.named_parameters():
            for regex in self.pipeline_config.trainable_params:
                pattern = re.compile(regex)
                if re.match(pattern, name):
                    if param.requires_grad:
                        param_list_["params"].append(param)
                        n_params += param.numel()

        param_list.append(param_list_)

        logging.info(f"Number of trainable parameters: {n_params}")

        optimizer_cls = getattr(
            importlib.import_module("torch.optim"),
            self.pipeline_config.optimizer_name,
        )
        optimizer = optimizer_cls(
            param_list, lr=lr, **self.pipeline_config.optimizer_kwargs
        )
        optimizers.append(optimizer)

        self.optims = optimizers
        schedulers_config = self.configure_lr_schedulers()

        for name, param in self.model.named_parameters():
            set_grad_false = True
            for regex in self.pipeline_config.trainable_params:
                pattern = re.compile(regex)
                if re.match(pattern, name):
                    if param.requires_grad:
                        set_grad_false = False
            if set_grad_false:
                param.requires_grad = False

        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logging.info(f"Number of trainable parameters: {num_trainable_params}")

        schedulers_config = self.configure_lr_schedulers()

        if schedulers_config is None:
            return optimizers

        return optimizers, [
            schedulers_config_ for schedulers_config_ in schedulers_config
        ]

    def configure_lr_schedulers(self):
        schedulers_config = []
        if self.pipeline_config.lr_scheduler_name is None:
            scheduler = None
            schedulers_config.append(scheduler)
        else:
            scheduler_cls = getattr(
                importlib.import_module("torch.optim.lr_scheduler"),
                self.pipeline_config.lr_scheduler_name,
            )
            scheduler = scheduler_cls(
                self.optims[0],
                **self.pipeline_config.lr_scheduler_kwargs,
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": self.pipeline_config.lr_scheduler_interval,
                "monitor": "val_loss",
                "frequency": self.pipeline_config.lr_scheduler_frequency,
            }
            schedulers_config.append(lr_scheduler_config)

        if all([scheduler is None for scheduler in schedulers_config]):
            return None

        return schedulers_config

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> dict:
        model_output = self.model(train_batch)
        logging.info(f"loss: {model_output['loss']}")

        return {
            "loss": model_output["loss"],
            "latent_recon_loss": model_output["latent_recon_loss"],
            "pixel_recon_loss": model_output["pixel_recon_loss"],
            "batch_idx": batch_idx,
        }

    def validation_step(self, val_batch: Dict[str, Any], val_idx: int) -> dict:
        model_output = self.model(val_batch, device=self.device)
        samples = self.model.log_samples(
            batch=val_batch,
            **self.log_samples_model_kwargs,
        )
        grid_samples = self.build_grids(
            val_batch, 
            samples, 
            mask_keys=["mask"] # TODO: do not hardcode this
        )
        other_metrics = self.compute_metrics(val_batch, samples) if self.metric_handler is not None else {}
        return {
            "metrics": {
                "loss": model_output["loss"],
                "latent_recon_loss": model_output["latent_recon_loss"],
                "pixel_recon_loss": model_output["pixel_recon_loss"],
                **other_metrics
            },
            "visuals": grid_samples
        }
    
    def build_grids(
            self, 
            batch: Dict[str, Any], 
            samples: Dict[str, Any], 
            mask_keys: list[str] | None = None,
            uid_key: str = "uid"
    ) -> Dict[str, Any]:
        output_logs: Dict[str, Any] = {}

        assert uid_key in batch, f"uid_key {uid_key} not in batch"

        all_images = {
            **{
                key: batch[key].to(self.device)
                for key in self.log_keys if key in batch
            },
            **{
                key: samples[key].to(self.device)
                for key in samples
            }
        }
        samples_keys = list(samples.keys())

        # renormalize masks from [0, 1] to [-1, 1]
        if mask_keys is not None:
            for key in mask_keys:
                if key in all_images:
                    all_images[key] = (all_images[key] - 0.5) * 2.0

        _, _, h, w = list(all_images.values())[0].shape
        shape = (1, 3, h, w)

        # we log one grid image per batch element
        # The grid contains:
        # - the input images (self.log_keys) in the first row (n_columns = len(self.log_keys))
        # - the samples (samples) in the second rows (n_columns = len(samples))
        # - the column-wise mix of the images in the third rows (n_columns = len(column_visualization_keys))

        n_columns = max(len(self.log_keys), len(samples), len(self.column_visualization_keys))
        batch_size = min([samples[k].shape[0] for k in samples])
        for image_index in range(0, batch_size):
            grid: list[list[torch.Tensor]] = []

            first_row = []
            for key_index in range(0, n_columns):
                if key_index < len(self.log_keys):
                    cell = all_images[self.log_keys[key_index]][image_index:image_index+1]
                else:
                    cell = torch.zeros(shape, device=self.device)  # pad with black images
                
                assert cell.shape == shape, f"Cell shape {cell.shape} does not match expected shape {shape}"
                first_row.append(cell)
            
            grid.append(first_row)

            second_row = []
            for key_index in range(0, n_columns):
                if key_index < len(samples_keys):
                    cell = all_images[samples_keys[key_index]][image_index:image_index+1]
                else:
                    cell = torch.zeros(shape, device=self.device)  # pad with black images
                
                assert cell.shape == shape, f"Cell shape {cell.shape} does not match expected shape {shape}"
                second_row.append(cell)

            grid.append(second_row)

            third_row = []
            for key_index in range(0, n_columns):
                if key_index < len(self.column_visualization_keys):
                    first_key, second_key = self.column_visualization_keys[key_index]
                    cell = mix_images_column_wise(
                        all_images[first_key][image_index:image_index+1],
                        all_images[second_key][image_index:image_index+1]
                    )
                else:
                    cell = torch.zeros(shape, device=self.device)  # pad with black images
                
                assert cell.shape == shape, f"Cell shape {cell.shape} does not match expected shape {shape}"
                third_row.append(cell)
            
            grid.append(third_row)

            
            image_key = batch[uid_key][image_index].replace("/", "_")
            output_logs[f"image_{image_key}"] = torch.cat(
                [torch.cat(row, dim=3) for row in grid], 
                dim=2
            )

        return output_logs

    def compute_metrics(self, batch: Dict[str, Any], samples: Dict[str, Any]) -> Dict[str, Any]:
        assert self.metric_handler is not None
        output_logs: Dict[str, Any] = {}

        for sample_key, sample in samples.items():
            metrics = self.metric_handler.compute(
                prediction=sample,
                target=batch[self.model.target_key]
            )
            for metric_key, value in metrics.items():
                output_logs[f"{metric_key}_{sample_key}"] = value
        return output_logs
