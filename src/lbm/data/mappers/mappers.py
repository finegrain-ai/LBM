from ast import Tuple
from typing import Any, Dict

from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import math
import hashlib
from .base import BaseMapper
from .masking import create_random_mask
from .aspect_ratios import get_target_size

from .mappers_config import (
    KeyRenameMapperConfig,
    RescaleMapperConfig,
    CustomResizeConfig,
    TorchvisionMapperConfig,
    RandomPixelMaskingConfig,
    RandomMaskConfig,
)
from torch import Tensor

class KeyRenameMapper(BaseMapper):
    """
    Rename keys in a sample according to a key map

    Args:

        config (KeyRenameMapperConfig): Configuration for the mapper

    Examples
    ########

    1. Rename keys in a sample according to a key map

    .. code-block:: python

        from cr.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

        config = KeyRenameMapperConfig(
            key_map={"old_key": "new_key"}
        )

        mapper = KeyRenameMapper(config)

        sample = {"old_key": 1}
        new_sample = mapper(sample)
        print(new_sample)  # {"new_key": 1}

    2. Rename keys in a sample according to a key map and a condition key

    .. code-block:: python

        from cr.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

        config = KeyRenameMapperConfig(
            key_map={"old_key": "new_key"},
            condition_key="condition",
            condition_fn=lambda x: x == 1
        )

        mapper = KeyRenameMapper(config)

        sample = {"old_key": 1, "condition": 1}
        new_sample = mapper(sample)
        print(new_sample)  # {"new_key": 1}

        sample = {"old_key": 1, "condition": 0}
        new_sample = mapper(sample)
        print(new_sample)  # {"old_key": 1}

    ```
    """

    def __init__(self, config: KeyRenameMapperConfig):
        super().__init__(config)
        self.key_map = config.key_map
        self.condition_key = config.condition_key
        self.condition_fn = config.condition_fn
        self.else_key_map = config.else_key_map

    def __call__(self, batch: Dict[str, Any], *args, **kwrags):
        if self.condition_key is not None:
            condition_key = batch[self.condition_key]
            if self.condition_fn(condition_key):
                for old_key, new_key in self.key_map.items():
                    if old_key in batch:
                        batch[new_key] = batch.pop(old_key)

            elif self.else_key_map is not None:
                for old_key, new_key in self.else_key_map.items():
                    if old_key in batch:
                        batch[new_key] = batch.pop(old_key)

        else:
            for old_key, new_key in self.key_map.items():
                if old_key in batch:
                    batch[new_key] = batch.pop(old_key)
        return batch


class TorchvisionMapper(BaseMapper):
    """
    Apply torchvision transforms to a sample

    Args:

        config (TorchvisionMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: TorchvisionMapperConfig):
        super().__init__(config)
        chained_transforms = []
        for transform, kwargs in zip(config.transforms, config.transforms_kwargs):
            transform = getattr(transforms, transform)
            chained_transforms.append(transform(**kwargs))
        self.transforms = transforms.Compose(chained_transforms)

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        if self.key in batch:
            batch[self.output_key] = self.transforms(batch[self.key])
        return batch


class RescaleMapper(BaseMapper):
    """
    Rescale a sample from [0, 1] to [-1, 1]

    Args:

        config (RescaleMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: RescaleMapperConfig):
        super().__init__(config)

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        if isinstance(batch[self.key], list):
            tmp = []
            for i, image in enumerate(batch[self.key]):
                tmp.append(2 * image - 1)
            batch[self.output_key] = tmp
        else:
            batch[self.output_key] = 2 * batch[self.key] - 1
        return batch

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
    
    def _seed_from_string(self, s: str) -> int:
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2**32)
    
    def _process(self, mask: Tensor, image: Tensor, seed: str | None) -> Tensor:
        if seed:
            generator = torch.Generator(device=image.device)
            generator.manual_seed(self._seed_from_string(seed))
        else:
            generator = None
        
        noise = torch.empty_like(image).uniform_(generator=generator)
        return image * (1 - mask) + noise * mask

class CustomResize(BaseMapper):
    """
    Crop the input so that its height and width are multiples of a given number.

    Args:

        config (CustomResizeConfig): Configuration for the mapper
    """

    def __init__(self, config: CustomResizeConfig):
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

    def _seed_from_string(self, s: str) -> int:
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2**32)
    
    def _process(self, image: Tensor, seed: int | str | None) -> Tensor:
        _, h, w = image.shape
        if isinstance(seed, str):
            seed = self._seed_from_string(seed)
        mask = create_random_mask((w, h), seed=seed)  # (width, height)
        mask = mask.unsqueeze(0) # 1, h, w
        if self.channels != 1:
            mask = mask.repeat(self.channels, 1, 1)  # n_channels, h, w
        return mask