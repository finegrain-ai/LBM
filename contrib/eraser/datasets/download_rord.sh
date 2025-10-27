#!/bin/bash
set -euo pipefail

mkdir -p data/presencesw
# Unofficial mirror, see https://huggingface.co/datasets/presencesw/RORD
hf download presencesw/RORD --repo-type dataset --include "zip/*" --local-dir ./data/presencesw/RORD

pushd data/presencesw/RORD/zip
# We must use 7z here with multithreading enabled
7z x RORD.zip -mmt=on -o../../RORD
popd