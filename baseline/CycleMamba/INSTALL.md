# CycleMamba Installation Guide

## Dependencies
CycleMamba requires the following special packages:
- Python 3.10.14
- PyTorch 2.1.1
- torchvision 0.15.2
- numpy 1.26.4
- causal_conv1d 1.1.0
- mamba_ssm 1.1.0

## Installation Steps

### 1. Install mamba_ssm
For installing mamba_ssm, please refer to [VMamba](https://github.com/MzeroMiko/VMamba) or [mamba](https://github.com/state-spaces/mamba).

### 2. Install causal_conv1d
```bash
pip install causal_conv1d==1.1.0
```

### 3. Alternative: Use pre-built wheels
If you encounter issues building from source, you can try pre-built wheels:
```bash
# For CUDA 11.8
pip install mamba_ssm-1.1.0+cu118-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.1.0+cu118-cp310-cp310-linux_x86_64.whl

# For CUDA 12.1
pip install mamba_ssm-1.1.0+cu121-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.1.0+cu121-cp310-cp310-linux_x86_64.whl
```

## Notes
- mamba_ssm requires CUDA toolkit to be installed
- The installation may take some time as it compiles CUDA kernels
- If you encounter compilation errors, make sure your CUDA version matches your PyTorch installation