#!/usr/bin/env bash
set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh
conda activate base

conda activate dcunet
jupyter notebook \
    --no-browser \
    --ip=0.0.0.0 \
    --allow-root \
    --NotebookApp.token=
