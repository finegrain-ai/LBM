import math
import random

import numpy as np
from .lama import MixedMaskGenerator
import torch

def build_mask_generator(image_size: tuple[int, int]) -> MixedMaskGenerator:
    """
    Build a MixedMaskGenerator from LaMa
    """

    av_resolution = math.sqrt(image_size[0] * image_size[1])
    # Params from https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/configs/training/data/abl-04-256-mh-dist.yaml
    # Used in lama-regular in https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/configs/training/lama-regular.yaml
    # Compared to official LaMa code, we have added 20% outpainting
    mask_gen_kwargs = {
        "irregular_proba": 0.4,
        "irregular_kwargs": {
            "max_angle": 4,
            "max_len": 200,
            "max_width": int(av_resolution / 256 * 100),
            "max_times": 5,
            "min_times": 1,
        },
        "box_proba": 0.4,
        "box_kwargs": {
            "margin": 10,
            "bbox_min_size": int(av_resolution / 256 * 30),
            "bbox_max_size": int(av_resolution / 256 * 150),
            "max_times": 4,
            "min_times": 1,
        },
        "segm_proba": 0,
        "outpainting_proba": 0.2,
    }

    return MixedMaskGenerator(**mask_gen_kwargs)


def create_random_mask(
    image_size: tuple[int, int],  # (width, height)
    mask_generator: MixedMaskGenerator | None = None,
    seed: int | str | None = None,
) -> torch.Tensor:
    """
    Wrapper around MixedMaskGenerator from LaMa

    The output mask is a torch.Tensor of shape (height, width) between 0 and 1
    """
    mask_generator = mask_generator or build_mask_generator(image_size)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    # mask_generator segmentation has been deactivated
    # because it's not used in official LaMa training
    # So we can use a fake image as input, only the size matters
    fake_image = np.zeros((3, image_size[1], image_size[0]), dtype=np.float32)
    mask = mask_generator(fake_image).squeeze(0)  # 1, h, w -> h, w
    return torch.from_numpy(mask)