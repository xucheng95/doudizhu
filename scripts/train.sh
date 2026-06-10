#!/bin/bash
set -e
cd "$(dirname "$0")/.."
if [ -f "venv/bin/activate" ]; then source venv/bin/activate; fi
python -m training.train --config configs/default.yaml "$@"
