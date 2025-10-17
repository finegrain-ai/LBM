# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma
import os
import random
import os.path as osp
import hashlib
import logging
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from lbm.contrib.data.sana.wids import ShardListDataset, ShardListDatasetMulti
from lbm.contrib.data.sana.aspect_ratios import get_aspect_ratios
import getpass
import torch.distributed as dist
from lbm.contrib.data.masking import create_random_mask

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


# Inspired from SanaWebDatasetMS
# https://github.com/NVlabs/Sana/blob/34b4aa5c102bfa36cb57ad68a43a15ec7fe3f411/diffusion/data/datasets/sana_data_multi_scale.py#L40-L41
# With following changes:
# 1. Remove vae/text feature loading related code
# 2. Remove caption related code
# 3. Change the logger
# 4. Use utils.get_aspect_ratios
# 5. Implement multi-images, one-extension per image, one transform per image
# 6. put max_retry as a parameter
# 7. Integrate with lama masking (for inpainter)
# 8. Use random pixel masking (for inpainter)

class InpainterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str | list[str],
        meta_path: None | str | list[str] = None, # if none, will try to find wids-meta.json in data_dir
        cache_dir: str = "data/cache/wids",
        max_shards_to_load: bool = None,
        resolution: int = 256,
        sort_dataset: bool = False,
        num_replicas: bool = None,
        max_retry: int = 10,
    ) -> None:

        self.logger = logging.getLogger(__name__)
        self.max_retry = max_retry

        self.base_size = resolution
        self.ratio_index = {}
        self.ratio_nums = {}
        # NB: SANA use LANCZOS for 2048 and 2880
        # It's not clear why they do that
        # See https://github.com/NVlabs/Sana/blob/34b4aa5c102bfa36cb57ad68a43a15ec7fe3f411/diffusion/data/datasets/sana_data_multi_scale.py#L97-L98
        self.interpolate_model = InterpolationMode.BICUBIC
        self.aspect_ratios = get_aspect_ratios(resolution)

        data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        meta_paths = meta_path if isinstance(meta_path, list) else [meta_path] * len(data_dirs)
        self.meta_paths = []
        for data_path, meta_path in zip(data_dirs, meta_paths):
            self.data_path = osp.expanduser(data_path)
            self.meta_path = osp.expanduser(meta_path) if meta_path is not None else None

            _local_meta_path = osp.join(self.data_path, "wids-meta.json")
            if meta_path is None and osp.exists(_local_meta_path):
                self.logger.info(f"loading from {_local_meta_path}")
                self.meta_path = meta_path = _local_meta_path

            if meta_path is None:
                self.meta_path = osp.join(
                    osp.expanduser(cache_dir),
                    self.data_path.replace("/", "--") + f".max_shards:{max_shards_to_load}" + ".wdsmeta.json",
                )

            assert osp.exists(self.meta_path), f"meta path not found in [{self.meta_path}] or [{_local_meta_path}]"
            self.logger.info(f"[SimplyInternal] Loading meta information {self.meta_path}")
            self.meta_paths.append(self.meta_path)

        self._initialize_dataset(num_replicas, sort_dataset)

        for k, v in self.aspect_ratios.items():
            self.ratio_index[float(k)] = []
            self.ratio_nums[float(k)] = 0
    
    def get_ratio_nums(self) -> dict[float, int]:
        return self.ratio_nums

    def get_aspect_ratios(self) -> dict[float, tuple[int, int]]:
        return self.aspect_ratios

    def _initialize_dataset(self, num_replicas: int | None, sort_dataset: bool) -> None:
        # uuid = abs(hash(self.meta_path)) % (10 ** 8)

        uuid = hashlib.sha256(self.meta_path.encode()).hexdigest()[:8]
        if len(self.meta_paths) > 0:
            self.dataset = ShardListDatasetMulti(
                self.meta_paths,
                cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"),
                sort_data_inseq=sort_dataset,
                num_replicas=num_replicas or dist.get_world_size(),
            )
        else:
            # TODO: tmp to ensure there is no bug
            self.dataset = ShardListDataset(
                self.meta_path,
                cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"),
            )
        self.ori_imgs_nums = len(self)
        self.logger.info(f"{self.dataset.data_info}")

    def getdata(self, idx: int) -> dict:
        data = self.dataset[idx]
        self.key = data["__key__"]
        info = data[".json"]

        data_info = {}
        ori_h, ori_w = info["height"], info["width"]

        # Calculate the closest aspect ratio and resize & crop image[w, h]
        closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratios)
        closest_size = list(map(lambda x: int(x), closest_size))
        # As in SANA, we store it in case of retry see __getitem__
        self.closest_ratio = closest_ratio

        img = data[".png"] if ".png" in data else data[".jpg"]

        if closest_size[0] / ori_h > closest_size[1] / ori_w:
            resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
        else:
            resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]

        img = data[".png"] if ".png" in data else data[".jpg"]

        img_transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB")),
                T.Resize(resize_size, interpolation=self.interpolate_model),  # Image.BICUBIC
                T.CenterCrop(closest_size),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        after = img_transform(img)

        # Generate an int seed from __key__
        seed = int(hashlib.sha256(self.key.encode("utf-8")).hexdigest(), 16) % (2**32)

        mask = create_random_mask(
            (closest_size[1], closest_size[0]),  # (width, height)
            seed=idx
        )

        # Following LBM eraser
        # Replace pixels (corresponding to a mask) in an image with uniformly sampled random values.
        noise = torch.empty_like(after).uniform_(
            generator=torch.Generator().manual_seed(seed)
        )
        before = after * (1 - mask) + noise * mask

        return {
            "img_hw": torch.tensor([ori_h, ori_w], dtype=torch.float32),
            "aspect_ratio": closest_ratio,
            "index": data["__index__"],
            "shard": "/".join(data["__shard__"].rsplit("/", 2)[-2:]),
            "shardindex": data["__shardindex__"],
            "before": before,
            "after": after,
            "uid": self.key,
        }

    def __getitem__(self, idx: int) -> dict:
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.ratio_index[self.closest_ratio])
        raise RuntimeError("Too many bad data.")

    def __len__(self) -> int:
        return len(self.dataset)

    def get_data_info(self, idx: int) -> dict | None:
        try:
            data = self.dataset[idx]
            info = data[".json"]
            key = data["__key__"]
            version = info.get("version", "others")
            return {"height": info["height"], "width": info["width"], "version": version, "key": key}
        except Exception as e:
            print(f"Error details: {str(e)}")
            return None

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    image_size = 256
    data_dir = ["data/Re-LAION-1300K/"]
    meta_path = ["data/Re-LAION-1300K/wids-meta-train.json"]
    train_dataset = InpainterDataset(data_dir=data_dir, meta_path=meta_path, resolution=image_size, num_replicas=1)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

    for data in tqdm(dataloader):
        # Test something here
        print(data["before"].shape)
        break
