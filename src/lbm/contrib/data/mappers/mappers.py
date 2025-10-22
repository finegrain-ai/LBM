from typing import Any, Dict
import torchvision.transforms.functional as F
import torch
from torch import Tensor
from lbm.data.mappers.base import BaseMapper
from lbm.trainer.utils import seed_from_string
from ..masking import create_random_mask
from ..aspect_ratios import get_target_size

from .mappers_config import (
    AspectRatioResizeConfig,
    RandomPixelMaskingConfig,
    RandomMaskConfig,
)

class RandomPixelMasking(BaseMapper):
    """
    Replace the pixels of an image within a given mask with random values
    Random values are sampled from a uniform distribution [0, 1]

    Args:
        config (RandomPixelMaskingConfig): Configuration for the mapper
    """

    def __init__(self, config: RandomPixelMaskingConfig):
        super().__init__(config)

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        batch[self.output_key] = self._process(
            mask=batch[self.config.mask_key],
            image=batch[self.config.key],
            seed=batch[self.config.seed_key] if self.config.seed_key else None,
        )
        return batch
    
    def _process(self, mask: Tensor, image: Tensor, seed: str | None) -> Tensor:
        if seed:
            generator = torch.Generator(device=image.device)
            generator.manual_seed(seed_from_string(seed))
        else:
            generator = None
        
        noise = torch.empty_like(image).uniform_(generator=generator)
        return image * (1 - mask) + noise * mask

class AspectRatioResize(BaseMapper):
    """
    Resize an image to a predefined list of size (hardcoded in aspect_ratios.py), depending on args.resolution 
    and the aspect ratio of the image.

    If config.size_output_key is not None, it outputs the size of the cropped image (useful for size-related bucketing).

    Args:

        config (AspectRatioResizeConfig): Configuration for the mapper
    """

    def __init__(self, config: AspectRatioResizeConfig):
        super().__init__(config)
        self.resolution = config.resolution
        self.size_output_key = config.size_output_key

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        if self.key in batch:
            cropped_image, size = self._process(batch[self.key])
            batch[self.output_key] = cropped_image
            if self.size_output_key:
                batch[self.size_output_key] = size
        return batch

    def _process(self, image: Tensor) -> tuple[Tensor, tuple[int, int]]:
        _, h, w = image.shape

        ori_ratio = w / h
        
        target_w, target_h = get_target_size(
            target_resolution=self.resolution,
            height=h,
            width=w
        )
        target_ratio = target_w / target_h
        if target_h != h or target_w != w:

            if ori_ratio > target_ratio:
                resize_h = target_h
                resize_w = int(ori_ratio * target_h)
            else:
                resize_w = target_w
                resize_h = int(target_w / ori_ratio)
            
            # We first downscale then center crop to avoid pixel stretching
            image = F.resize(image, (resize_h, resize_w))
            image = F.center_crop(image, (target_h, target_w))
        return image, (target_h, target_w)

class RandomMask(BaseMapper):
    """
    Creates a random mask, lama-like

    Args:
        config (RandomMaskConfig): Configuration for the mapper
    """
    def __init__(self, config: RandomMaskConfig):
        super().__init__(config)
        self.channels = config.channels

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        batch[self.output_key] = self._process(
            image=batch[self.config.key],
            seed=batch[self.config.seed_key] if self.config.seed_key else None,
        )
        return batch
    
    def _process(self, image: Tensor, seed: int | str | None) -> Tensor:
        _, h, w = image.shape
        if isinstance(seed, str):
            seed = seed_from_string(seed)
        mask = create_random_mask((w, h), seed=seed)  # (width, height)
        mask = mask.unsqueeze(0)  # 1, h, w
        if self.channels != 1:
            mask = mask.repeat(self.channels, 1, 1)  # n_channels, h, w
        return mask