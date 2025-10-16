import datetime
import logging
import os
import random
import re
import shutil
from typing import List, Optional, Tuple, Any, Dict
from torch import distributed as dist

import braceexpand
import fire
import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import grad_norm
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.transforms import InterpolationMode
from pytorch_lightning.utilities import rank_zero_only

from torchmetrics import MetricCollection
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics import Metric
from torchvision.utils import make_grid
import torch.distributed as dist

from lbm.data.datasets import DataModule, DataModuleConfig
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
    RandomPixelMasking,
    RandomPixelMaskingConfig,
)
from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.utils import StateDictAdapter
from dataclasses import field
from neptune.types import File
from torch.optim.optimizer import Optimizer

class EraserLogger(Callback):
    """
    Eraser Logger Callback made for Neptune.

    Log LPIPS, DISTS, PSNR metrics, in addition to train and val losses.

    It uses Lightning's `self.log` method to log metrics, so 
    accumulation/distributed logging is handled by lightning.

    Args:
        num_steps (list[int]): List of number of steps to log metrics for.
    """
    def __init__(
        self, 
        num_steps: list[int],
    ):
        super().__init__()
        self.num_steps = num_steps
        self.device = None  # delay initialization
        self.metrics = None  # delay initialization
    
    def setup(self, trainer: Trainer, pl_module: TrainingPipeline, stage=None) -> None:
        assert isinstance(trainer.logger, loggers.NeptuneLogger)

        self.device = pl_module.device

        metrics: dict[str, Metric] = {}
        for n in self.num_steps:
            metrics[f"lpips_{n}"] = LearnedPerceptualImagePatchSimilarity().to(self.device)
            metrics[f"dists_{n}"] = DeepImageStructureAndTextureSimilarity().to(self.device)
            metrics[f"psnr_{n}"] = PeakSignalNoiseRatio((-1.0, 1.0)).to(self.device)

        self.metrics = MetricCollection(metrics).to(self.device)
    
    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.log("train/loss", outputs["loss"])
    
    @rank_zero_only
    def on_before_optimizer_step(self, trainer: Trainer, pl_module: TrainingPipeline, optimizer: Optimizer) -> None:
        total_grad_norm = grad_norm(trainer.model, norm_type=2)["grad_2.0_norm_total"]
        self.log("grad_norm", total_grad_norm, on_step=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.log("val/loss", outputs["loss"], on_epoch=True, sync_dist=True)

        # Infer the samples
        samples = pl_module.log_samples(batch)

        samples_by_steps = {
            step: samples[f"samples_{step}_steps"] 
            for step in self.num_steps 
            if f"samples_{step}_steps" in samples
        }

        # Run the metrics        
        for metric_key in self.metrics:
            # predictions in lbm_model.py are named f"samples_{num_step}_steps"
            n_steps = int(metric_key.split("_")[-1])
            pred = samples_by_steps.get(n_steps, None)
            if pred is None:
                logging.warning(f"Key f'samples_{n_steps}_steps' not found in outputs. Skipping metric {metric_key}.")
                continue
            pred = pred.clamp(-1, 1)
            metric = self.metrics[metric_key]
            gt = batch[pl_module.model.target_key]

            metric.update(pred.to(self.device), gt.to(self.device))
        
        # Visualize the images
        grids = self._build_grids(
            batch, 
            samples_by_steps=samples_by_steps, 
            log_keys=pl_module.log_keys,
            source_key=pl_module.model.source_key,
            uid_key="uid"
        )

        gathered_imgs = self._gather_dict_on_rank0(grids)

        if dist.get_rank() == 0:
            assert gathered_imgs is not None
            for image_key in gathered_imgs.keys():
                image = gathered_imgs[image_key]
                # rescale from [-1, 1] to [0, 1]
                image = (image / 2 + 0.5).clamp(0, 1)
                neptune_file = File.as_image(image.cpu().squeeze().permute(1, 2, 0).clip(0, 1))
                # only compatible with neptune logger
                trainer.logger.experiment[f"val/{image_key}"].append(neptune_file, step=trainer.global_step)

    def _gather_dict_on_rank0(self, local_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not dist.is_initialized():
            return local_dict  # single-process case

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Each process sends its dict
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_dict, gathered, dst=0)

        if rank == 0:
            # Merge into a single dict of Tensors
            merged = {}
            for d in gathered:
                for k, v in d.items():
                    if k not in merged:
                        merged[k] = v.cpu()
                    else:
                        logging.warning(f"Key {k} already exists in merged dict. Dropping duplicate.")
            return merged
        else:
            return None

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline
    ):
        output = self.metrics.compute()
        for key, value in output.items():
            self.log(f"val/{key}", value, sync_dist=True)
        self.metrics.reset()
    
    def _build_grids(
        self, 
        batch: Dict[str, Any], 
        samples_by_steps: Dict[int, Any], 
        log_keys: List[str],
        source_key: str,
        uid_key: str,
    ) -> Dict[str, Any]:
        """
        Build a 3 rows visualization grid for each image in the batch.
        1st row: input images (log_keys)
        2nd row: predicted images (for each num_steps)
        3rd row: column-wise mixed predicted and input images (for each num_steps)
        """
        output_grids: Dict[str, Any] = {}
        assert uid_key in batch, f"uid_key {uid_key} not in batch"

        batch_size, _, h, w = list(samples_by_steps.values())[0].shape
        shape = (1, 3, h, w)

        nrow = max(len(log_keys), len(self.num_steps))
        
        for image_index in range(0, batch_size):
            grid_image: list[torch.Tensor] = []

            first_row = []
            for key_index in range(0, nrow):
                if key_index < len(log_keys):
                    cell = batch[log_keys[key_index]][image_index:image_index+1].to(self.device)
                else:
                    cell = torch.zeros(shape, device=self.device)  # pad with black images

                assert cell.shape == shape, f"Cell shape {cell.shape} does not match expected shape {shape}"
                first_row.append(cell)

            grid_image.extend(first_row)

            second_row = []
            third_row = []
            for key_index in range(0, nrow):
                if key_index < len(self.num_steps):
                    pred = samples_by_steps[self.num_steps[key_index]][image_index:image_index+1].to(self.device)
                    mix = self._mix_images_column_wise(pred, batch[source_key][image_index:image_index+1]).to(self.device)
                else:
                    pred = torch.zeros(shape, device=self.device)  # pad with black images
                    mix = torch.zeros(shape, device=self.device)  # pad with black images

                assert pred.shape == shape, f"Cell shape {pred.shape} does not match expected shape {shape}"
                second_row.append(pred)
                third_row.append(mix)

            grid_image.extend(second_row)
            grid_image.extend(third_row)

            grid = torch.cat(grid_image, dim=0)

            image_key = batch[uid_key][image_index].replace("/", "_")
            output_grids[f"image_{image_key}"] = make_grid(
                grid,
                nrow=nrow
            )

        return output_grids
    
    def _mix_images_column_wise(self, a: torch.Tensor, b: torch.Tensor, column_width: int = 5) -> torch.Tensor:
        """
        Mix two batched images column-wise along width.

        Args:
            a (torch.Tensor): Tensor of shape (b, c, h, w)
            b (torch.Tensor): Tensor of shape (b, c, h, w)
            column_width (int): Width (in pixels) of each alternating column block.

        Returns:
            torch.Tensor: Mixed tensor of shape (b, c, h, w)
        """
        assert a.shape == b.shape, "Inputs must have the same shape"
        assert a.ndim == 4, "Inputs must be 4D tensors (b, c, h, w)"
        _, _, _, w = a.shape


        device = a.device
        dtype = a.dtype


        b = b.to(device=device, dtype=dtype)


        # Build mask pattern [1, 1, 0, 0, 1, 1, 0, 0, ...] along width dimension
        mask = torch.zeros(w, dtype=torch.bool, device=device)
        toggle = True
        for x in range(0, w, column_width):
            x_end = min(x + column_width, w)
            mask[x:x_end] = toggle
            toggle = not toggle


        # with shape (1, 1, 1, w) it broadcasts mask to (b, c, h, w)
        mask = mask.view(1, 1, 1, w)


        return torch.where(mask, a, b)

def get_model(
    backbone_signature: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    source_key: str = "before_masked",
    target_key: str = "after",
    bridge_noise_sigma: float = 0.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
):

    conditioners = []

    # Load pretrained model as base
    pipe = StableDiffusionPipeline.from_pretrained(
        backbone_signature,
        torch_dtype=torch.bfloat16,
    )

    ### MMMDiT ###
    # Get Architecture
    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,  # Add downsampled_image
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            # SD15 specific
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=[
            # SD15 specific
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
        ],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280, 1280], # match SD1.5
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280, 1280], # match SD1.5 = block_out_channels
        transformer_layers_per_block=1,
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=8,
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=False,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=None,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch.bfloat16)

    state_dict = pipe.unet.state_dict()

    denoise_state_dict = denoiser.state_dict()

    # Adapt the shapes
    state_dict_adapter = StateDictAdapter()
    state_dict = state_dict_adapter(
        model_state_dict=denoise_state_dict,
        checkpoint_state_dict=state_dict,
        regex_keys=[
            r"class_embedding.linear_\d+.(weight|bias)",
            r"conv_in.weight",
            r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
            r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
        ],
        strategy="zeros",
    )
    
    denoiser.load_state_dict(state_dict, strict=True)

    del pipe

    if conditioning_images_keys != [] or conditioning_masks_keys != []:

        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

    # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()
    vae.to(torch.bfloat16)

    # LBM Config
    config = LBMConfig(
        ucg_keys=None,
        source_key=source_key,
        target_key=target_key,
        mask_key=None, # for the eraser all the pixels in the target image are valid
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )

    # LBM Model
    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)

    return model

def get_filter_mappers(
    image_size: Tuple[int, int], # (height, width)
    resize_mode: str = "Resize" # CenterCrop or Resize
) -> list[MapperWrapper | KeyFilter]:

    match resize_mode:
        case "CenterCrop":
            resize_args = {
                "size": image_size,
            }
        case "Resize":
            resize_args = {
                "size": image_size,
                "interpolation": InterpolationMode.NEAREST_EXACT,
            }
        case _:
            raise ValueError(f"Unsupported resize_mode: {resize_mode}")
        
        
    filters_mappers = [
        KeyFilter(KeyFilterConfig(keys=["before.jpg", "after.jpg", "mask.png", "__key__"], verbose=True)),
        MapperWrapper(
            [
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            "before.jpg": "before",
                            "after.jpg": "after",
                            "mask.png": "mask",
                            "__key__": "uid",
                        }
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="before",
                        transforms=["ToTensor", resize_mode],
                        transforms_kwargs=[
                            {},
                            resize_args,
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="after",
                        transforms=["ToTensor", resize_mode],
                        transforms_kwargs=[
                            {},
                            resize_args,
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="mask",
                        transforms=["ToTensor", resize_mode, "Normalize"],
                        transforms_kwargs=[
                            {},
                            resize_args,
                            {"mean": 0.0, "std": 1.0},
                        ],
                    )
                ),
                # Random pixel masking is made on [0, 1] tensors (before RescaleMapper)
                RandomPixelMasking(
                    RandomPixelMaskingConfig(
                        key="before",
                        mask_key="mask",
                        output_key="before_masked",
                        verbose=True,
                        seed_key="uid"
                    )
                ),
                RescaleMapper(RescaleMapperConfig(key="mask", verbose=True)), # for visualization only
                RescaleMapper(RescaleMapperConfig(key="before", verbose=True)), # for visualization only
                RescaleMapper(RescaleMapperConfig(key="before_masked", verbose=True)),
                RescaleMapper(RescaleMapperConfig(key="after", verbose=True)),

            ],
        ),
    ]

    return filters_mappers


def get_data_module(
    train_shards: List[str],
    validation_shards: List[str],
    batch_size: int,
    image_size: Tuple[int, int], # (height, width)
    resize_mode: str = "Resize" # CenterCrop or Resize
):

    # TRAIN
    train_filters_mappers = get_filter_mappers(image_size, resize_mode)

    # unbrace urls
    train_shards_path_or_urls_unbraced = []
    for train_shards_path_or_url in train_shards:
        train_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(train_shards_path_or_url)
        )

    # shuffle shards
    random.shuffle(train_shards_path_or_urls_unbraced)

    # data config
    train_data_config = DataModuleConfig(
        shards_path_or_urls=train_shards_path_or_urls_unbraced,
        decoder="pil",
        # RORD dataset contains 400K images in 400 shards
        # 200 out of 400 shards
        shuffle_before_split_by_node_buffer_size=min(200, len(train_shards_path_or_urls_unbraced)),
        # Each node has 400/4 ~ 100 shards, so 30 looks fine
        shuffle_before_split_by_workers_buffer_size=50,
        # 10 workers means each worker sees ~ 10 shards = 10k samples
        # Set it to 4k
        shuffle_before_filter_mappers_buffer_size=4000,
        # not needed to shuffle after filter mappers
        shuffle_after_filter_mappers_buffer_size=None,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    # VALIDATION
    validation_filters_mappers = get_filter_mappers(image_size, resize_mode)

    # unbrace urls
    validation_shards_path_or_urls_unbraced = []
    for validation_shards_path_or_url in validation_shards:
        validation_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(validation_shards_path_or_url)
        )

    validation_data_config = DataModuleConfig(
        shards_path_or_urls=validation_shards_path_or_urls_unbraced,
        decoder="pil",
        # deactivate shuffling for validation, so we validate on the same
        # samples each time
        shuffle_before_split_by_node_buffer_size=None,
        shuffle_before_split_by_workers_buffer_size=None,
        shuffle_before_filter_mappers_buffer_size=None,
        shuffle_after_filter_mappers_buffer_size=None,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    # data module
    data_module = DataModule(
        train_config=train_data_config,
        train_filters_mappers=train_filters_mappers,
        eval_config=validation_data_config,
        eval_filters_mappers=validation_filters_mappers,
    )

    return data_module


def main(
    train_shards: List[str] = ["pipe:cat path/to/train/shards"],
    validation_shards: List[str] = ["pipe:cat path/to/validation/shards"],
    backbone_signature: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    source_key: str = "before_masked",
    target_key: str = "after",
    neptune_project: str = "LBM-Eraser",
    batch_size: int = 8,
    num_steps: List[int] = [1, 2, 4],
    learning_rate: float = 5e-5,
    learning_rate_scheduler: str = None,
    learning_rate_scheduler_kwargs: dict = {},
    optimizer: str = "AdamW",
    optimizer_kwargs: dict = {},
    timestep_sampling: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    selected_timesteps: List[float] = None,
    prob: List[float] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    config_yaml: dict = None,
    save_ckpt_path: str = "./checkpoints",
    gradient_clip_val: Optional[float] = None,
    log_interval: int = 10,
    resume_from_checkpoint: bool = True,
    max_epochs: int = 100,
    bridge_noise_sigma: float = 0.005,
    save_interval: int = 1000,
    limit_val_batches: int = 2,
    val_check_interval: int = 1000,
    save_top_k: int = 1,
    path_config: str = None,
    image_size: Tuple[int, int] | List[int] = (480, 640),  # (H, W)
    resize_mode: str = "Resize" # CenterCrop or Resize
):
    model = get_model(
        backbone_signature=backbone_signature,
        vae_num_channels=vae_num_channels,
        unet_input_channels=unet_input_channels,
        source_key=source_key,
        target_key=target_key,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        pixel_loss_type=pixel_loss_type,
        latent_loss_type=latent_loss_type,
        latent_loss_weight=latent_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
        selected_timesteps=selected_timesteps,
        prob=prob,
        conditioning_images_keys=conditioning_images_keys,
        conditioning_masks_keys=conditioning_masks_keys,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    if isinstance(image_size, list):
        assert len(image_size) == 2, "image_size must be a tuple of (height, width)"
        image_size = tuple(image_size)

    data_module = get_data_module(
        train_shards=train_shards,
        validation_shards=validation_shards,
        batch_size=batch_size,
        image_size=image_size,
        resize_mode=resize_mode,
    )

    train_parameters = ["denoiser.*"]

    # Training Config
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["before", "after", "mask"],
        trainable_params=train_parameters,
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={
            "input_shape": None,
            "num_steps": num_steps,
        },
    )
    if (
        os.path.exists(save_ckpt_path)
        and resume_from_checkpoint
        and "last.ckpt" in os.listdir(save_ckpt_path)
    ):
        start_ckpt = f"{save_ckpt_path}/last.ckpt"
        print(f"Resuming from checkpoint: {start_ckpt}")

    else:
        start_ckpt = None

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config, verbose=True)

    pipeline.save_hyperparameters(
        {
            f"embedder_{i}": embedder.config.to_dict()
            for i, embedder in enumerate(model.conditioner.conditioners)
        }
    )

    pipeline.save_hyperparameters(
        {
            "denoiser": model.denoiser.config,
            "vae": model.vae.config.to_dict(),
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
            "training_noise_scheduler": model.training_noise_scheduler.config,
            "sampling_noise_scheduler": model.sampling_noise_scheduler.config,
        }
    )

    training_signature = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "-LBM-Eraser"
    )
    dir_path = f"{save_ckpt_path}/logs/{training_signature}"
     # if not initialized (single gpu), we are on the main process before ddp spawns
    if not dist.is_initialized():
        os.makedirs(dir_path, exist_ok=True)
        if path_config is not None:
            shutil.copy(path_config, f"{save_ckpt_path}/config.yaml")
    run_name = training_signature

    # Ignore parameters unused during training
    ignore_states = []
    for name, param in pipeline.model.named_parameters():
        ignore = True
        for regex in ["denoiser."]:
            pattern = re.compile(regex)
            if re.match(pattern, name):
                ignore = False
        if ignore:
            ignore_states.append(param)

    # FSDP Strategy
    strategy = FSDPStrategy(
        auto_wrap_policy=ModuleWrapPolicy(
            [
                UNet2DConditionModel,
                BasicTransformerBlock,
                ResnetBlock2D,
                torch.nn.Conv2d,
            ]
        ),
        activation_checkpointing_policy=ModuleWrapPolicy(
            [
                BasicTransformerBlock,
                ResnetBlock2D,
            ]
        ),
        sharding_strategy="SHARD_GRAD_OP",
        ignored_states=ignore_states,
    )
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise ValueError("No GPU available for training.")

    trainer = Trainer(
        log_every_n_steps=log_interval,
        gradient_clip_val=gradient_clip_val,
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        default_root_dir="logs",
        logger=loggers.NeptuneLogger(
            project=neptune_project, name=run_name, log_model_checkpoints=False
        ),
        callbacks=[
            EraserLogger(
                num_steps=num_steps
            ),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                every_n_train_steps=save_interval,
                save_last=True,
                save_top_k=save_top_k,
                monitor="val/loss",       # metric to rank by
                mode="min",
            ),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        max_epochs=max_epochs,
    )

    trainer.fit(pipeline, data_module, ckpt_path=start_ckpt)


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
