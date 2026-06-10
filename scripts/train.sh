#!/bin/bash
set -e
cd "$(dirname "$0")/.."
/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9 -m training.train --config configs/default.yaml "$@"
