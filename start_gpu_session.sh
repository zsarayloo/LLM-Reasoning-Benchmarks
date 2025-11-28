#!/bin/bash

# Script to start GPU session with conda environment activated
# Usage: ./start_gpu_session.sh

echo "Starting GPU session with conda environment..."
echo "Available GPU partitions:"
sinfo -p gpu_a100,gpu_h100,gpu_l40s,gpu_p100 -o "%20P %20N %10c %10m %25f %10G %6t"

echo ""
echo "Starting interactive session on H100 GPU..."

# Start GPU session and automatically activate conda environment
srun -p gpu_h100 --gres=gpu:1 --time=4:00:00 --pty bash -c "
    echo 'Connected to GPU node: '$(hostname)
    echo 'Activating conda environment...'
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate /work/zsaraylo/LLM-Reasoning-Benchmarks/.condaenv
    echo 'Conda environment activated!'
    echo 'Current directory: '$(pwd)
    echo 'Testing GPU access...'
    python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')\" 2>/dev/null || echo 'PyTorch not available - install with: pip install torch'
    echo ''
    echo 'You are now on a GPU node with conda environment activated!'
    echo 'Run your experiments with: python src/experiment/run_livemathbench_cot_mistral.py'
    echo ''
    exec bash
"
