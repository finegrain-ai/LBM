import importlib
import logging
import re
import time
from typing import Any, Dict

import pytorch_lightning as pl
import torch

from ..models.base.base_model import BaseModel
from .training_config import TrainingConfig
from .utils import seed_from_string

logging.basicConfig(level=logging.INFO)


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
        loss = model_output["loss"]
        logging.info(f"loss: {loss}")
        return {
            "loss": loss,
            "batch_idx": batch_idx,
        }

    def validation_step(self, val_batch: Dict[str, Any], val_idx: int) -> dict:
        if self.model.config.seed_key is not None:
            assert self.model.config.seed_key in val_batch
            seed = seed_from_string("-".join(val_batch[self.model.config.seed_key]))
        else:
            seed = None
        if self.model.timestep_sampling == "custom_timesteps":
            target_key = self.model.config.target_key
            n_samples = val_batch[target_key].shape[0]
            selected_timesteps = self.model.selected_timesteps
            prob = self.model.prob

            out = {}

            # Inspired by what is done in SD3 https://arxiv.org/pdf/2403.03206 
            #  Section 5.3.3
            #  "we sample loss levels equidistant in t ∈ (0, 1) and compute validation"
            #  "loss for each level separately. We then average the loss across all but the last (t = 1) levels"
            # NB: In SD3, t = 1 corresponds to timestep 1000 here
            #
            # Here we follow the same idea with key differences : 
            # * We intuit that they skip t = 1000 in SD3 because t2i is an ambiguous task
            #   where different output images can correspond to the same input prompt.
            #   In the context of i2i, the input image provides a strong conditioning
            #   signal and we can include t = 1000 in the validation loss
            # * We use levels from `selected_timesteps` and weight them by `prob` as in training

            for timestep in selected_timesteps:
                timesteps_tensor = torch.full(
                    (n_samples,), timestep, device=self.device, dtype=torch.long
                )
                timestep_out = self.model(val_batch, device=self.device, seed=seed, timestep=timesteps_tensor)
                out[f"loss_t{timestep}"] = timestep_out["loss"]
                out[f"latent_recon_loss_t{timestep}"] = timestep_out["latent_recon_loss"]
                out[f"pixel_recon_loss_t{timestep}"] = timestep_out["pixel_recon_loss"]
            
            out["loss"] = sum(
                out[f"loss_t{timestep}"] * prob[i]
                for i, timestep in enumerate(selected_timesteps)
            )

            out["latent_recon_loss"] = sum(
                out[f"latent_recon_loss_t{timestep}"] * prob[i]
                for i, timestep in enumerate(selected_timesteps)
            )

            out["pixel_recon_loss"] = sum(
                out[f"pixel_recon_loss_t{timestep}"] * prob[i]
                for i, timestep in enumerate(selected_timesteps)
            )

        else: # model.timestep_sampling != "custom_timesteps"
            
            forward_out = self.model(val_batch, device=self.device, seed=seed)
            out["loss"] = forward_out["loss"]
            out["latent_recon_loss"] = forward_out["latent_recon_loss"]
            out["pixel_recon_loss"] = forward_out["pixel_recon_loss"]
        
        out["metrics"] = self.model.compute_metrics(val_batch)

        return out

    def log_samples(self, batch: Dict[str, Any]):
        logging.debug("log_samples")
        logs = self.model.log_samples(
            batch,
            **self.log_samples_model_kwargs,
        )

        if logs is not None:
            N = min([logs[keys].shape[0] for keys in logs])
        else:
            N = 0

        # Log inputs
        if self.log_keys is not None:
            for key in self.log_keys:
                if key in batch:
                    if N > 0:
                        logs[key] = batch[key][:N]
                    else:
                        logs[key] = batch[key]

        return logs
