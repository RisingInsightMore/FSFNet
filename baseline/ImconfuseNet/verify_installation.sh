#!/bin/bash
set -e

echo "=== ImconfuseNet Installation Verification ==="

ERRORS=0

echo "Checking Python..."
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    ERRORS=$((ERRORS + 1))
else
    echo "OK: Python found"
fi

echo "Checking PyTorch..."
if ! python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null; then
    echo "ERROR: PyTorch not installed"
    ERRORS=$((ERRORS + 1))
else
    echo "OK: PyTorch installed"
fi

echo "Checking CUDA..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "WARNING: CUDA not available (CPU mode will be used)"
else
    echo "OK: CUDA available"
fi

echo "Checking dataset symlinks..."
for dataset in AVIID DayDrone NightDrone; do
    if [ -L "datasets/$dataset" ]; then
        if [ -d "datasets/$dataset" ]; then
            echo "OK: datasets/$dataset symlink exists and points to directory"
        else
            echo "ERROR: datasets/$dataset symlink exists but target not found"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo "ERROR: datasets/$dataset symlink not found"
        ERRORS=$((ERRORS + 1))
    fi
done

echo "Checking training script..."
if [ -f "run_train.sh" ]; then
    echo "OK: run_train.sh exists"
else
    echo "ERROR: run_train.sh not found"
    ERRORS=$((ERRORS + 1))
fi

echo "Checking testing script..."
if [ -f "run_test.sh" ]; then
    echo "OK: run_test.sh exists"
else
    echo "ERROR: run_test.sh not found"
    ERRORS=$((ERRORS + 1))
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "=== Verification PASSED ==="
    exit 0
else
    echo "=== Verification FAILED with $ERRORS errors ==="
    exit 1
fi
