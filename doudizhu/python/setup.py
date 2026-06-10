"""Setup script for doudizhu environment.

Build C++ extension first, then install Python package.

Quick install:
    cd doudizhu
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j4
    cd ../python
    pip install -e .
"""
from setuptools import setup, find_packages
import os

# Try to find the compiled .so or .pyd
import sys
import glob

# Check if pybind11 is available
try:
    import pybind11
    has_pybind11 = True
except ImportError:
    has_pybind11 = False

setup(
    name="doudizhu",
    version="1.0.0",
    description="Doudizhu (Fight the Landlord) RL Environment",
    author="Reasonix Code",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
)
