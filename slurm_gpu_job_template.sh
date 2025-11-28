#!/bin/bash
#SBATCH --job-name=llm_reasoning
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load your conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path if needed
conda activate your_env_name  # Replace with your actual environment name

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/tmp/transformers_cache_$SLURM_JOB_ID
export HF_HOME=/tmp/hf_home_$SLURM_JOB_ID

# Print GPU information
echo "=== GPU Information ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check if GPU is available in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Navigate to project directory
cd /work/zsaraylo/LLM-Reasoning-Benchmarks

# Run your experiment (replace with actual script)
echo "Starting experiment..."
python src/experiment/run_livemathbench_cot_mistral.py

# Clean up temporary directories
rm -rf /tmp/transformers_cache_$SLURM_JOB_ID
rm -rf /tmp/hf_home_$SLURM_JOB_ID

echo "Job completed!"
