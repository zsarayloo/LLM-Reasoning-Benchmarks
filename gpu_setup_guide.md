# GPU Setup Guide for LLM Reasoning Benchmarks

## Available GPU Nodes

Based on your SLURM cluster, you have access to these GPU partitions:

| Partition | Node | GPUs | Memory | Status | Recommendation |
|-----------|------|------|--------|--------|----------------|
| gpu_h100 | gpu-pr1-06 | 2x H100 | 515GB | **IDLE** | ⭐ **BEST CHOICE** |
| gpu_h100 | gpu-pr1-03 | 4x H100 | 1TB | MIX | High-end option |
| gpu_a100 | gpu-pr1-02 | 4x A100 | 1TB | MIX | Good alternative |
| gpu_l40s | gpu-pr1-04 | 3x L40S | 773GB | MIX | Mid-range |
| gpu_l40s | gpu-pr1-05 | 2x L40S | 515GB | MIX | Mid-range |
| gpu_p100 | gpu-pr1-01 | 4x P100 | 128GB | IDLE | Budget option |

## Quick Start Commands

### 1. Check GPU Status
```bash
./run_gpu_experiments.sh status
```

### 2. Start Interactive GPU Session
```bash
# Recommended: H100 node (idle)
./run_gpu_experiments.sh interactive gpu_h100 1 4:00:00

# Alternative: P100 node (also idle)
./run_gpu_experiments.sh interactive gpu_p100 1 4:00:00
```

### 3. Submit Batch Job
```bash
# Copy and modify the template
cp slurm_gpu_job_template.sh my_experiment.sh
# Edit my_experiment.sh with your conda env name and experiment
sbatch my_experiment.sh
```

## Step-by-Step Setup

### Step 1: Make Scripts Executable
```bash
chmod +x run_gpu_experiments.sh
chmod +x slurm_gpu_job_template.sh
```

### Step 2: Check Current Status
```bash
./run_gpu_experiments.sh status
```

### Step 3: Start Interactive Session (Recommended for Testing)
```bash
# Start on H100 node (best performance, currently idle)
srun -p gpu_h100 --gres=gpu:1 --time=4:00:00 --pty bash

# Once in the GPU node:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate your_env_name  # Replace with your actual env name
cd /work/zsaraylo/LLM-Reasoning-Benchmarks

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Step 4: Run Your Experiments
```bash
# Test with Mistral model
python src/experiment/run_livemathbench_cot_mistral.py

# Test with Llama model
python src/experiment/run_livemathbench_cot_llama.py
```

## Model Memory Requirements

Your models will use these approximate GPU memory amounts:

- **Mistral-7B-Instruct** (FP16): ~14GB VRAM
- **Llama-3.2-3B-Instruct** (FP16): ~6GB VRAM

All your available GPU nodes have sufficient memory for these models.

## Batch Job Template Usage

1. Copy the template:
```bash
cp slurm_gpu_job_template.sh run_mistral_experiment.sh
```

2. Edit the script:
```bash
nano run_mistral_experiment.sh
```

3. Modify these lines:
```bash
conda activate your_actual_env_name  # Line 15
python src/experiment/run_livemathbench_cot_mistral.py  # Line 32
```

4. Submit the job:
```bash
sbatch run_mistral_experiment.sh
```

5. Monitor the job:
```bash
squeue -u $USER
tail -f logs/slurm_JOBID.out
```

## Troubleshooting

### GPU Not Detected
If `torch.cuda.is_available()` returns `False`:
1. Check you're on a GPU node: `echo $SLURM_JOB_NODELIST`
2. Check GPU allocation: `echo $CUDA_VISIBLE_DEVICES`
3. Verify PyTorch CUDA installation: `python -c "import torch; print(torch.version.cuda)"`

### Out of Memory Errors
1. Use smaller batch sizes in your experiments
2. Enable gradient checkpointing
3. Use FP16 precision (already enabled in your model loaders)

### Job Queue Issues
1. Check queue status: `squeue`
2. Use idle nodes first (gpu-pr1-06 or gpu-pr1-01)
3. Reduce time limit if needed

## Your Model Loaders Are GPU-Ready

Your existing model loaders already support GPU:
- `model/mistral_7b_loader.py` ✅
- `model/lama3_3b_loader.py` ✅

Both automatically detect CUDA and use FP16 on GPU, FP32 on CPU.
