from typing import Any, Callable, Dict, List, Optional

from pydantic.dataclasses import dataclass

from lbm.data.mappers.mappers_config import BaseMapperConfig

@dataclass
class RandomPixelMaskingConfig(BaseMapperConfig):
    """
    Replace pixels (corresponding to a mask) in an image with uniformly sampled random values.

    Args:

        key (str): Key to apply the masking to.
        mask_key (str): Key containing the mask to apply.
        seed_key (str | None): Key containing the seed to use for random number generation.
    """

    key: str = "image"
    mask_key: str = "mask"
    seed_key: str | None = None

@dataclass
class AspectRatioResizeConfig(BaseMapperConfig):
    """
    Resize an image to a predefined list of size (hardcoded in aspect_ratios.py), depending on args.resolution 
    and the aspect ratio of the image.

    If size_output_key is not None, it outputs the size of the cropped image (useful for size-related bucketing).

    Args:

        key (str): Key to apply the cropping to.
        size_output_key (Optional[str]): Key to store the resulting size (height, width) of the cropped image.
            If None, the size is not stored. Default is None.
        resolution (int): The target resolution to resize the cropped image to 
    """

    key: str = "image"
    size_output_key: str = "image_size"
    resolution: int = 256

@dataclass
class RandomMaskConfig(BaseMapperConfig):
    """
    Creates a random mask, lama-like

    Args:

        key (str): Key to create the mask for, used to get the image size.
        seed_key (Optional[str]): Key containing the seed to use for random number generation.
            If None, no seed is used. Default is None.
        channels (int): Number of channels of the mask to create. Default is 1.
    """

    key: str = "image"
    seed_key: Optional[str] = None
    channels: int = 1
