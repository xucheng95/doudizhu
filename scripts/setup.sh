#!/bin/bash
# One-command project setup: creates venv, installs deps, builds C++ extension
set -e

cd "$(dirname "$0")/.."
echo "=== Doudizhu Environment Setup ==="

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
fi

source venv/bin/activate

# 2. Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt

# 3. Build C++ extension (.so)
echo "Building C++ extension..."
PYBIND11_INC=$(python -c "import pybind11; print(pybind11.get_include())")
PYTHON_INC=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")

OS="$(uname)"
if [ "$OS" = "Darwin" ]; then
    c++ -O3 -shared -std=c++17 \
      -I doudizhu/include \
      -I "$PYBIND11_INC" \
      -I "$PYTHON_INC" \
      -o doudizhu/python/doudizhu/doudizhu_cpp.so \
      doudizhu/src/cards.cpp doudizhu/src/game.cpp \
      doudizhu/src/env.cpp doudizhu/pybind/bindings.cpp \
      -undefined dynamic_lookup \
      -fvisibility=hidden
else
    PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_LIBDIR=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    c++ -O3 -shared -std=c++17 -fPIC \
      -I doudizhu/include \
      -I "$PYBIND11_INC" \
      -I "$PYTHON_INC" \
      -o doudizhu/python/doudizhu/doudizhu_cpp.so \
      doudizhu/src/cards.cpp doudizhu/src/game.cpp \
      doudizhu/src/env.cpp doudizhu/pybind/bindings.cpp \
      -L"$PY_LIBDIR" -lpython${PY_VER} \
      -fvisibility=hidden
fi

echo ""
echo "=== Setup Complete ==="
echo "Activate:   source venv/bin/activate"
echo "Train:      python -m training.train --config configs/default.yaml"
echo "TensorBoard: tensorboard --logdir runs"
