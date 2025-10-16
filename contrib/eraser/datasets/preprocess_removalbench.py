# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
#     "pillow",
#     "tqdm",
# ]
# ///

import argparse
import sys
import tarfile
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import snapshot_download
from tqdm import tqdm


def shard_indices(n: int, k: int) -> list[tuple[int, int]]:
    """Split n items into k shards, as evenly as possible."""
    base, rem = divmod(n, k)
    result = []
    start = 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        result.append((start, end))
        start = end
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default="data/BaiLing")
    parser.add_argument("--repo_id", type=str, default="BaiLing/RemovalBench")
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--prefix", type=str, default="RemovalBench")
    parser.add_argument("--force_redownload", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id}...")
    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["images/*.png", "gt/*.png", "masks/*.png"],
        force_download=args.force_redownload,
        local_dir_use_symlinks=False,
    )

    images_dir = Path(local_dir) / "images"
    gt_dir = Path(local_dir) / "gt"
    masks_dir = Path(local_dir) / "masks"

    if not all(p.is_dir() for p in [images_dir, gt_dir, masks_dir]):
        print("❌ Expected subfolders 'images', 'gt', and 'masks' not found.", file=sys.stderr)
        sys.exit(1)

    # Match files by stem
    image_stems = {p.stem for p in images_dir.glob("*.png")}
    gt_stems = {p.stem for p in gt_dir.glob("*.png")}
    mask_stems = {p.stem for p in masks_dir.glob("*.png")}

    common = sorted(image_stems & gt_stems & mask_stems)
    print(f"✅ Found {len(common)} matched PNG samples.")

    triples = [
        (images_dir / f"{s}.png", gt_dir / f"{s}.png", masks_dir / f"{s}.png", s)
        for s in common
    ]

    for i, (start, end) in enumerate(shard_indices(len(triples), args.num_shards)):
        if start == end:
            continue

        
        shard_path = args.out_dir / f"{args.prefix}-{i:04d}.tar"
        print(f"🧩 Writing {shard_path} with {end - start} samples...")

        with tarfile.open(shard_path, "w", dereference=True) as tar:
            for idx, (img, gt, mask, stem) in enumerate(triples[start:end], start=start + 1):
                base = str(idx)
                tar.add(img, arcname=f"{base}-{stem}.before.png")
                tar.add(gt, arcname=f"{base}-{stem}.after.png")
                tar.add(mask, arcname=f"{base}-{stem}.mask.png")

    print("🎉 Done creating shards!")


if __name__ == "__main__":
    main()
