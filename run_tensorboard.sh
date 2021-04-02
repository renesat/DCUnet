#!/usr/bin/env bash
set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh
conda activate base

conda activate dcunet
tensorboard --host=0.0.0.0 --logdir="/workdir/results/tb"
