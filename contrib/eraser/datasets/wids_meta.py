#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import tarfile


def expand_brace_pattern(pattern: str) -> list[str]:
    """Expand Bash-style brace expressions like imagenet-train-{000000..000003}.tar."""
    try:
        result = subprocess.check_output(["bash", "-c", f"echo {pattern}"], text=True)
        return result.strip().split()
    except Exception as e:
        raise RuntimeError(f"Failed to expand pattern {pattern}: {e}")


def md5sum(filename: str, chunk_size: int=8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def count_samples_in_tar(tar_path: str) -> int:
    """Count number of unique sample prefixes (stems) in a tarfile, following WebDataset conventions."""
    stems = set()
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if "." not in name:
                continue
            stem = name.split(".")[0]
            stems.add(stem)
    return len(stems)


def main():
    parser = argparse.ArgumentParser(description="Generate WIDS shard index JSON from tar shards.")
    parser.add_argument("--tars", required=True, help="Tar file pattern, e.g., 'imagenet-train-{000000..000003}.tar'")
    parser.add_argument("--output", help="Output path for wids-meta.json (default: parent of first tar)")
    parser.add_argument("--name", required=True, help="Name of the dataset (e.g., 'imagenet-train').")
    args = parser.parse_args()

    # Expand brace pattern
    files = expand_brace_pattern(args.tars)
    if not files:
        raise ValueError(f"No files matched pattern {args.tars}")

    shardlist = []

    for fpath in files:
        if not os.path.exists(fpath):
            print(f"⚠️  Warning: file not found: {fpath}")
            continue

        filesize = os.path.getsize(fpath)
        md5 = md5sum(fpath)
        nsamples = count_samples_in_tar(fpath)

        shardlist.append({
            "url": os.path.basename(fpath),
            "md5sum": md5,
            "nsamples": nsamples,
            "filesize": filesize,
        })

        print(f"{os.path.basename(fpath)}: {nsamples} samples, {filesize} bytes, md5={md5}")

    wids_index = {
        "__kind__": "wids-shard-index-v1",
        "wids_version": 1,
        "shardlist": shardlist,
        "name": args.name,
    }

    # Save to <parent of first tar>/wids-meta.json
    parent_dir = os.path.dirname(os.path.abspath(files[0])) or "."
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(parent_dir, "wids-meta.json")
    
    with open(output_path, "w") as f:
        json.dump(wids_index, f, indent=2)

    print(f"\n✅ WIDS index written to {output_path} ({len(shardlist)} shards)")


if __name__ == "__main__":
    main()
