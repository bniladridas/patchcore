# MLX + Anomalib Setup Guide

## Installation

```bash
# Clone and setup environment
cd anomalib
pip install -r requirements.txt

# For MLX on Apple Silicon
pip install mlx

# Install anomalib from source
pip install git+https://github.com/openvinotoolkit/anomalib.git
```

## Running the Notebook

```bash
jupyter notebook mlx_anomalib_notebook.ipynb
```

## Notes

- MLX requires macOS with Apple Silicon (M1/M2/M3)
- MLX is currently in preview; API may change
- For best performance, use PyTorch MPS backend as fallback
