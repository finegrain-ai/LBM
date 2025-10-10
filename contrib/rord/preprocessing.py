# Inspired from https://github.com/Forty-lock/RORD/blob/3736a6ba0520e0eac78f966f8788f252735d065c/preprocessing.py
# * Code revamped
# * Build finemask from polygon annotations of dynamic objects
# * rescaling low definition images
# * Remove the poisson blending

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL.Image import Image
from tqdm import tqdm

def build_fine_mask(
    json_data: dict, coarse_mask_path: Path, width: int, height: int, threshold: float | None
) -> tuple[Image, int, int]:
    dic = json_data["Learning_Data_Info."]["Annotation"]
    resolution = json_data["Raw_Data_Info."]["Resolution"].split(", ")
    w, h = int(resolution[0]), int(resolution[1])
    scale_w = w / width
    scale_h = h / height
    assert abs(scale_w - scale_h) < 1e-6, f"Non-uniform scaling not supported: {scale_w} vs {scale_h}"
    scale = scale_w

    fine_mask = np.zeros((height, width), np.uint8)
    n_discarded = 0
    n_kept = 0

    if threshold is not None:
        assert coarse_mask_path is not None
        assert coarse_mask_path.exists(), f"coarse mask not found: {coarse_mask_path}"
        # In practice, RORD provides the inverted coarse mask
        inverted_coarse_mask = cv2.imread(coarse_mask_path, cv2.IMREAD_GRAYSCALE) // 255
        assert inverted_coarse_mask is not None, f"coarse mask not found: {coarse_mask_path}"
    else:
        inverted_coarse_mask = None

    for dic_ann in dic:
        polys = dic_ann["segmentation"]
        class_id = dic_ann["Class_ID"]
        # The json contains 2 types of class_ids
        # - "F*" classes are the ones we want to keep - for example F38 is human
        # - "BO*", classes are for background or others - for example F30 is table
        # see the full list of label https://github.com/Forty-lock/RORD
        if class_id[0] != "F":
            continue
        if len(polys) < 6:
            continue

        polys = np.array(polys).reshape(-1, 2)
        # rescale
        polys = (polys / scale).astype(np.int32)

        poly_mask = cv2.fillPoly(np.zeros((height, width), np.uint8), [polys], 1)

        if poly_mask.sum() <= 1:
            continue

        if threshold is not None:
            assert inverted_coarse_mask is not None
            # Discard masks that are not inside the coarse mask
            pixel_outside_of_coarse_mask = inverted_coarse_mask * poly_mask
            percent_outside = np.sum(pixel_outside_of_coarse_mask) / np.sum(poly_mask)
            if percent_outside > threshold:
                n_discarded = n_discarded + 1
                continue
            else:
                n_kept = n_kept + 1

        fine_mask = poly_mask + fine_mask * (1 - poly_mask)

    fine_mask = fine_mask * 255
    assert fine_mask.shape == (height, width), f"{fine_mask.shape} vs {(height, width)}"
    return PIL.Image.fromarray(fine_mask), n_kept, n_discarded


def build_shard(
    output_path: Path,
    samples: list[Path],
    width: int,
    height: int,
    n_sample_per_shard: int = 1000,
    threshold: float | None = 0.02,
) -> None:
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    n_kept_total = 0
    n_discarded_total = 0
    for before in samples:
        sample_name = before.stem
        gt_paths = list(before.parent.glob("*G0001.jpg"))
        assert len(gt_paths) == 1, f"Expected one GT image for {before}, found {len(gt_paths)}"
        after_path = gt_paths[0]
        assert after_path.exists(), f"GT image not found for {before}"

        json_path = before.parent.parent.parent / "label" / f"{before.parent.name}" / f"{sample_name}.json"
        assert json_path.exists(), f"JSON file not found for {json_path}"

        with PIL.Image.open(before) as img:
            width, height = img.size
            assert img.size == (width, height), f"Image size mismatch for {before}"

        fine_mask, mask_n_kept, mask_n_discarded = build_fine_mask(
            json_data=json.load(Path.open(json_path, encoding="UTF8")),
            coarse_mask_path=before.parent.parent.parent / "mask" / f"{before.parent.name}" / f"{sample_name}_M.png",
            width=width,
            height=height,
            threshold=threshold,
        )
        n_kept_total += mask_n_kept
        n_discarded_total += mask_n_discarded
        # Now save before, after, mask to a webdatset compatible structure
        fine_mask.save(output_path / f"{sample_name}.mask.png")
        shutil.copy(before, output_path / f"{sample_name}.before.jpg")
        shutil.copy(after_path, output_path / f"{sample_name}.after.jpg")

    percent_kept = (
        (n_kept_total / (n_kept_total + n_discarded_total)) * 100 if (n_kept_total + n_discarded_total) > 0 else 0
    )
    print(f"Built shard at {output_path} with {len(samples)} samples, ({percent_kept:.2f}% kept)")

def process_split(
    base_path: Path,
    split: str,
    output_path: Path,
    width: int,
    height: int,
    force: bool,
    threshold: float | None,
    n_sample_per_shard: int = 1000,
):
    split_path = base_path / split
    scene_list = Path(split_path / "img").glob("*")
    shards: list[list[Path]] = []
    sample_list = [p for scene in scene_list for p in scene.glob("*.jpg") if not str(p).endswith("G0001.jpg")]
    print(f"Found {len(sample_list)} samples in {split} split")
    shards = [sample_list[i : i + n_sample_per_shard] for i in range(0, len(sample_list), n_sample_per_shard)]
    print(f"Sharded into {len(shards)} shards, with up to {n_sample_per_shard} samples each")
    n_leading_zeros = max(6, len(str(len(shards))))
    for shard_idx, shard in enumerate(tqdm(shards)):
        shard_name = f"{split}-{shard_idx:0{n_leading_zeros}d}"
        # handle resuming if the tar file already exists
        if (output_path / f"{shard_name}.tar").exists():
            if not force:
                print(f"Shard {shard_name} already exists, skipping")
                continue
            else:
                print(f"Shard {shard_name} already exists, but force is True, rebuilding")
                shutil.rmtree(output_path / shard_name, ignore_errors=True)

        shard_path = output_path / shard_name
        build_shard(shard_path, shard, width, height, n_sample_per_shard, threshold)
        # compress shard to a .tar file and delete the uncompressed folder
        os.system(f"tar --sort=name -cf {shard_path}.tar -C {output_path} {shard_name}")  # noqa: S605
        shutil.rmtree(shard_path)


def main() -> None:
    data_path = Path("data/RORD")
    width = 960
    height = 540
    threshold = 0.02
    n_sample_per_shard = 1000
    output_path = Path("data/RORD-processed")
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        process_split(
            base_path=data_path,
            split=split,
            output_path=output_path,
            width=width,
            height=height,
            threshold=threshold,
            n_sample_per_shard=n_sample_per_shard,
            force=False,
        )


if __name__ == "__main__":
    main()
