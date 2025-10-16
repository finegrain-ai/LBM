from typing import Any, Dict, Tuple

from numpy import size
from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import hashlib
from .base import BaseMapper
from .mappers_config import (
    KeyRenameMapperConfig,
    RescaleMapperConfig,
    TorchvisionMapperConfig,
    RandomPixelMaskingConfig,
    ResizeAndCenterCropConfig
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

class ResizeAndCenterCrop(BaseMapper):
    """
    Resize/Center-Crop the image so we do not stretch the pixel.

    Args:
        config (ResizeAndCenterCropConfig): Configuration for the mapper
    """

    def __init__(self, config: ResizeAndCenterCropConfig):
        super().__init__(config)

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        batch[self.output_key] = self._process(
            image=batch[self.config.key],
            size=self.config.image_size,
            interpolation=self.config.interpolation,
        )
        return batch

    def _process(
        self, 
        image: Tensor, 
        size: Tuple[int, int], # h, w
        interpolation: transforms.InterpolationMode
    ) -> Tensor:
        target_height, target_width = size
        target_ratio = target_width / target_height
        image_width, image_height = F.get_image_size(image)
        image_ratio = image_width / image_height
        if image_ratio > target_ratio:
            # Image is wider than target ratio, resize based on height
            new_height = target_height
            new_width = int(target_height * image_ratio)
        else:
            # Image is taller than target ratio, resize based on width
            new_width = target_width
            new_height = int(target_width / image_ratio)

        image = F.resize(image, size=(new_height, new_width), interpolation=self.config.interpolation)
        return F.center_crop(image, size)