from typing import Any, Callable, Dict, List, Optional

from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class BaseMapperConfig(BaseConfig):
    """
    Base configuration for mappers.

    Args:

        verbose (bool):
            If True, print debug information. Defaults to False

        key (Optional[str]):
            Key to apply the mapper to. Defaults to None

        output_key (Optional[str]):
            Key to store the output of the mapper. Defaults to None
    """

    verbose: bool = False
    key: Optional[str] = None
    output_key: Optional[str] = None


@dataclass
class KeyRenameMapperConfig(BaseMapperConfig):
    """
    Rename keys in a sample according to a key map

    Args:

        key_map (Dict[str, str]): Dictionary with the old keys as keys and the new keys as values
        condition_key (Optional[str]): Key to use for the condition. Defaults to None
        condition_fn (Optional[Callable[[Any], bool]]): Function to use for the condition to be met so
            the key map is applied. Defaults to None.
        else_key_map (Optional[Dict[str, str]]): Dictionary with the old keys as keys and the new keys as values
            if the condition is not met. Defaults to None *i.e.* the original key will be used.
    """

    key_map: Dict[str, str] = None
    condition_key: Optional[str] = None
    condition_fn: Optional[Callable[[Any], bool]] = None
    else_key_map: Optional[Dict[str, str]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.key_map is not None, "key_map should be provided"
        assert all(
            isinstance(old_key, str) and isinstance(new_key, str)
            for old_key, new_key in self.key_map.items()
        ), "key_map should be a dictionary with string keys and values"
        if self.condition_key is not None:
            assert self.condition_fn is not None, "condition_fn should be provided"
            assert callable(self.condition_fn), "condition_fn should be callable"
        if self.condition_fn is not None:
            assert self.condition_key is not None, "condition_key should be provided"
            assert isinstance(
                self.condition_key, str
            ), "condition_key should be a string"
        if self.else_key_map is not None:
            assert all(
                isinstance(old_key, str) and isinstance(new_key, str)
                for old_key, new_key in self.else_key_map.items()
            ), "else_key_map should be a dictionary with string keys and values"


@dataclass
class TorchvisionMapperConfig(BaseMapperConfig):
    """
    Apply torchvision transforms to a sample

    Args:

        key (str): Key to apply the transforms to
        transforms (torchvision.transforms): List of torchvision transforms to apply
        transforms_kwargs (Dict[str, Any]): List of kwargs for the transforms
    """

    key: str = "image"
    transforms: List[str] = None
    transforms_kwargs: List[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.transforms is None:
            self.transforms = []
        if self.transforms_kwargs is None:
            self.transforms_kwargs = []
        assert len(self.transforms) == len(
            self.transforms_kwargs
        ), "Number of transforms and kwargs should be same"


@dataclass
class RescaleMapperConfig(BaseMapperConfig):
    """
    Rescale a sample from [0, 1] to [-1, 1]

    Args:

        key (str): Key to rescale
    """

    key: str = "image"

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

class CropModConfig(BaseMapperConfig):
    """
    Crop the input so that its height and width are multiples of a given number.

    Args:

        key (str): Key to apply the cropping to.
        mod (int): The multiple to crop to. Default is 8.
        size_output_key (Optional[str]): Key to store the resulting size (height, width) of the cropped image.
            If None, the size is not stored. Default is None.
    """

    key: str = "image"
    mod: int = 8
    size_output_key: Optional[str] = None

class RandomAspectRatioConfig(BaseMapperConfig):
    """
    Randomly change the aspect ratio of an image by a factor sampled from a given list of aspect ratios.

    Args:

        keys (list[str]): Keys to apply the aspect ratio change to.
        output_keys (Optional[list[str]]): 
            Keys to store the resulting images with changed aspect ratios
            If None, the output will overwrite the input keys. Default is None.
        aspect_ratios (list[float | None]): 
            List of aspect ratio (w/h) factors to sample from.
            If None is included in the list, the original aspect ratio may be kept.
        seed_key (Optional[str]): Key containing the seed to use for random number generation.
        weights (Optional[list[float]]):
            Weights for sampling the aspect ratios. If None, uniform sampling is used. Default is None.
    """

    keys: list[str] = None
    output_keys: Optional[list[str]] = None
    aspect_ratios: list[float | None] = None
    seed_key: Optional[str] = None
    weights: Optional[list[float]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.keys is not None and len(self.keys) > 0, "keys must be a non-empty list"
        if self.output_keys is not None:
            assert len(self.output_keys) == len(self.keys), "output_keys must have the same length as keys"
        if self.aspect_ratios is None or len(self.aspect_ratios) == 0:
            raise ValueError("aspect_ratios must be a non-empty list")
        assert all(
            ar is None or (isinstance(ar, (float, int)) and ar > 0)
            for ar in self.aspect_ratios
        ), "aspect_ratios must be a list of positive numbers or None"

        if self.weights is not None:
            assert len(self.weights) == len(self.aspect_ratios), "weights must have the same length as aspect_ratios"
            assert all(w >= 0 for w in self.weights), "weights must be non-negative"
            if sum(self.weights) == 0:
                raise ValueError("At least one weight must be positive")

        if self.seed_key is not None:
            assert isinstance(self.seed_key, str), "seed_key must be a string"