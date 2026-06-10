#!/bin/bash
# Build the doudizhu C++ extension
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Use system python 3.9 (with framework)
PYTHON="python3"
PYBIND11_INC="$($PYTHON -c "import pybind11; print(pybind11.get_include())")"
PYTHON_INC="/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers"

echo "PYBIND11_INC: $PYBIND11_INC"
echo "PYTHON_INC: $PYTHON_INC"

c++ -O3 -march=native -shared -std=c++17 \
  -I "$PROJECT_DIR/include" \
  -I "$PYBIND11_INC" \
  -I "$PYTHON_INC" \
  -o "$PROJECT_DIR/python/doudizhu/doudizhu_cpp.so" \
  "$PROJECT_DIR/src/cards.cpp" \
  "$PROJECT_DIR/src/game.cpp" \
  "$PROJECT_DIR/src/env.cpp" \
  "$PROJECT_DIR/pybind/bindings.cpp" \
  -F/Library/Developer/CommandLineTools/Library/Frameworks \
  -framework Python3 \
  -fvisibility=hidden

echo "Build complete: python/doudizhu/doudizhu_cpp.so"
