#!/bin/bash

# GPU Experiment Runner for LLM Reasoning Benchmarks
# This script helps you submit jobs to different GPU partitions

# Function to display available GPU nodes
show_gpu_status() {
    echo "=== Available GPU Nodes ==="
    echo "A100 (4 GPUs, 1TB RAM): gpu-pr1-02 [mix]"
    echo "H100 (4 GPUs, 1TB RAM): gpu-pr1-03 [mix]"
    echo "H100 (2 GPUs, 515GB RAM): gpu-pr1-06 [idle] ⭐ RECOMMENDED"
    echo "L40S (3 GPUs, 773GB RAM): gpu-pr1-04 [mix]"
    echo "L40S (2 GPUs, 515GB RAM): gpu-pr1-05 [mix]"
    echo "P100 (4 GPUs, 128GB RAM): gpu-pr1-01 [idle]"
    echo ""
    echo "Current status:"
    sinfo -p gpu_a100,gpu_h100,gpu_l40s,gpu_p100 -o "%20P %20N %10c %10m %25f %10G %6t"
}

# Function to submit interactive GPU job
interactive_gpu() {
    local partition=$1
    local gpus=${2:-1}
    local time=${3:-"4:00:00"}
    
    echo "Starting interactive session on $partition with $gpus GPU(s)..."
    srun -p $partition --gres=gpu:$gpus --time=$time --pty bash
}

# Function to submit batch job
submit_batch() {
    local script=$1
    local partition=$2
    local gpus=${3:-1}
    local time=${4:-"12:00:00"}
    local job_name=${5:-"llm_reasoning"}
    
    echo "Submitting batch job: $script to $partition"
    sbatch -p $partition --gres=gpu:$gpus --time=$time --job-name=$job_name $script
}

# Main menu
case "$1" in
    "status")
        show_gpu_status
        ;;
    "interactive")
        # Usage: ./run_gpu_experiments.sh interactive gpu_h100 1 4:00:00
        interactive_gpu $2 $3 $4
        ;;
    "batch")
        # Usage: ./run_gpu_experiments.sh batch script.sh gpu_h100 1 12:00:00 job_name
        submit_batch $2 $3 $4 $5 $6
        ;;
    *)
        echo "Usage: $0 {status|interactive|batch}"
        echo ""
        echo "Examples:"
        echo "  $0 status                                    # Show GPU status"
        echo "  $0 interactive gpu_h100 1 4:00:00          # Start interactive session"
        echo "  $0 batch run_mistral.sh gpu_h100 1 12:00:00 mistral_job"
        echo ""
        show_gpu_status
        ;;
esac
