#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python3 -m training.train --config configs/default.yaml "$@"
