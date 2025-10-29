# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.26.4",
#     "opencv-python>=4.11.0.86",
#     "pillow>=11.3.0",
#     "tqdm>=4.67.1",
# ]
# ///
# Inspired from https://github.com/Forty-lock/RORD/blob/3736a6ba0520e0eac78f966f8788f252735d065c/preprocessing.py
# * Code revamped
# * Build finemask from polygon annotations of dynamic objects
# * rescaling low definition images
# * Remove the poisson blending

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import PIL
from PIL.Image import Image
from tqdm import tqdm


@dataclass(frozen=True)
class SampleRecord:
    before_path: Path
    original_stem: str
    output_stem: str


def _allocate_random_id(rng: random.Random, used_ids: set[str], width: int = 6) -> str:
    max_ids = 10**width
    if len(used_ids) >= max_ids:
        raise ValueError(
            f"Cannot allocate more than {max_ids} unique random identifiers with width {width}."
        )
    while True:
        candidate = f"{rng.randrange(max_ids):0{width}d}"
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate

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
        inverted_coarse_mask = cv2.imread(str(coarse_mask_path), cv2.IMREAD_GRAYSCALE) // 255
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
    samples: list[SampleRecord],
    width: int,
    height: int,
    n_sample_per_shard: int = 1000,
    threshold: float | None = 0.02,
) -> None:
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    n_kept_total = 0
    n_discarded_total = 0
    for sample in samples:
        before = sample.before_path
        sample_name = sample.original_stem
        output_stem = sample.output_stem

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
        fine_mask.save(output_path / f"{output_stem}.mask.png")
        shutil.copy(before, output_path / f"{output_stem}.before.jpg")
        shutil.copy(after_path, output_path / f"{output_stem}.after.jpg")

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
    shuffle: bool = False,
    rng: Optional[random.Random] = None,
    used_random_ids: Optional[set[str]] = None,
):
    split_path = base_path / split
    scene_list = Path(split_path / "img").glob("*")
    sample_paths = [p for scene in scene_list for p in scene.glob("*.jpg") if not str(p).endswith("G0001.jpg")]
    print(f"Found {len(sample_paths)} samples in {split} split")

    sample_records: list[SampleRecord] = []
    for before in sample_paths:
        original_stem = before.stem
        if shuffle:
            assert rng is not None, "Random generator must be provided when shuffle is enabled."
            assert used_random_ids is not None, "Shared id registry must be provided when shuffle is enabled."
            random_id = _allocate_random_id(rng, used_random_ids)
            output_stem = f"{random_id}_{original_stem}"
        else:
            output_stem = original_stem
        sample_records.append(SampleRecord(before_path=before, original_stem=original_stem, output_stem=output_stem))
    
    if shuffle:
        # reorder samples by the randomly-prefixed generated output_stem
        sample_records.sort(key=lambda r: r.output_stem)

    shards = [
        sample_records[i : i + n_sample_per_shard] for i in range(0, len(sample_records), n_sample_per_shard)
    ]

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess the RORD dataset into a webdataset format.")
    parser.add_argument("--data-path", type=Path, default=Path("data/RORD"), help="Root directory of the raw dataset.")
    parser.add_argument(
        "--output-path", type=Path, default=Path("data/RORD-shuffled"), help="Destination directory for shards."
    )
    parser.add_argument("--width", type=int, default=960, help="Expected width of the processed images.")
    parser.add_argument("--height", type=int, default=540, help="Expected height of the processed images.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Maximum allowed percentage of fine mask area outside the coarse mask before discarding.",
    )
    parser.add_argument(
        "--n-sample-per-shard",
        type=int,
        default=1000,
        help="Maximum number of samples per shard.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild shards even if the corresponding tar files already exist.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_false",
        dest="shuffle",
        help=(
            "Keep original frame stems when writing the webdataset samples "
            "instead of assigning random identifiers."
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Optional random seed to make shuffled identifiers deterministic.",
    )
    parser.set_defaults(shuffle=True)
    args = parser.parse_args(argv)

    data_path = args.data_path
    width = args.width
    height = args.height
    threshold = args.threshold
    n_sample_per_shard = args.n_sample_per_shard
    output_path = args.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.shuffle_seed) if args.shuffle else None
    used_random_ids: set[str] | None = set() if args.shuffle else None

    for split in ["train", "val"]:
        process_split(
            base_path=data_path,
            split=split,
            output_path=output_path,
            width=width,
            height=height,
            threshold=threshold,
            n_sample_per_shard=n_sample_per_shard,
            force=args.force,
            shuffle=args.shuffle,
            rng=rng,
            used_random_ids=used_random_ids,
        )


if __name__ == "__main__":
    main()