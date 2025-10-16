#!/usr/bin/env python3
import os
import re
import tarfile
import random
import shutil
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
DATASET_DIR = Path("./data/RORD-processed/")  # directory containing val-*.tar
OUTPUT_DIR = Path("./data/RORD-processed/")
TMP_EXTRACT_DIR = Path("./data/RORD-processed-tmp/")
MAX_SAMPLES_PER_SHARD = 32  # ~340 total scenes, it gives 11 shards
PATTERN = "val-0*.tar"  # pattern to match tar files

# --- Step A: List tar files ---
tar_files = sorted(DATASET_DIR.glob(PATTERN))
print(f"Found {len(tar_files)} tar files.")

# --- Step B: Build mapping scene → frames ---
scene_to_frames = defaultdict(list)
frame_pattern = re.compile(r"I-(\d+_[A-Z0-9]+_(T|W)\d+)_F(\d+)")

for tar_path in tar_files:
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".jpg") and not member.name.endswith(".png"):
                continue
            m = frame_pattern.search(os.path.basename(member.name))
            if m:
                scene_id, _, frame_id = m.groups()
                prefix = f"I-{scene_id}_F{frame_id}"
                scene_to_frames[scene_id].append((tar_path, prefix))
            else:
                print(f"Warning: Could not parse frame from {member.name} in {tar_path}")

print(f"Found {len(scene_to_frames)} unique scenes.")

# --- Step C: Randomly select one frame per scene ---
selected_frames = {}
for scene, frames in scene_to_frames.items():
    selected_frames[scene] = random.choice(frames)
print(f"Selected {len(selected_frames)} frames (one per scene).")

# --- Step D: Extract selected frames (3 files each) ---
if TMP_EXTRACT_DIR.exists():
    shutil.rmtree(TMP_EXTRACT_DIR)
TMP_EXTRACT_DIR.mkdir(parents=True)

for i, (scene, (tar_path, prefix)) in enumerate(selected_frames.items(), 1):
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if prefix in m.name]
        tar.extractall(path=TMP_EXTRACT_DIR, members=members)
    if i % 100 == 0:
        print(f"Extracted {i}/{len(selected_frames)} samples...")

print("Extraction complete.")

# --- Step E: Re-tar into new shards ---
def make_tar_shard(shard_idx, files):
    shard_path = OUTPUT_DIR / f"val-lite-{shard_idx:06d}.tar"
    with tarfile.open(shard_path, "w") as tar:
        for f in files:
            # remove the top-level directory (e.g., "val-000050/")
            arcname = f.name  # just keep the filename
            tar.add(f, arcname=arcname)
    print(f"Created {shard_path} with {len(files)//3} samples (flat layout)")

# Gather all extracted files grouped by sample prefix
all_samples = defaultdict(list)
for f in TMP_EXTRACT_DIR.rglob("*.*"):
    if match := frame_pattern.search(f.name):
        prefix = f"I-{match.group(1)}_F{match.group(2)}"
        all_samples[prefix].append(f)

samples = list(all_samples.values())
samples.sort(key=lambda s: s[0].name)

OUTPUT_DIR.mkdir(exist_ok=True)
for shard_idx in range(0, len(samples), MAX_SAMPLES_PER_SHARD):
    batch = samples[shard_idx:shard_idx+MAX_SAMPLES_PER_SHARD]
    batch_files = [f for s in batch for f in s]
    make_tar_shard(shard_idx // MAX_SAMPLES_PER_SHARD, batch_files)

shutil.rmtree(TMP_EXTRACT_DIR)

print("✅ Done! New dataset created in", OUTPUT_DIR)
