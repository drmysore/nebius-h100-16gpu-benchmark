#!/bin/bash
# multinode.sh - Run 16-GPU multi-node training with a single command
# Usage: source multinode.sh [exercise1|exercise2] [optimized]
#
# Examples:
#   source multinode.sh                    # Run Exercise 1 training (default config)
#   source multinode.sh exercise1          # Run Exercise 1 training
#   source multinode.sh exercise1 optimized # Run Exercise 1 with optimized config
#   source multinode.sh exercise2          # Run Exercise 2 benchmarks

# =============================================================================
# Configuration
# =============================================================================
MASTER_ADDR="10.2.0.129"
WORKER_ADDR="10.2.0.0"
MASTER_PORT="29500"
NNODES=2
NPROC_PER_NODE=8
REPO_PATH="/home/supreethlab/repos/nebius-h100-16gpu-benchmark"

# Parse arguments
EXERCISE="${1:-exercise1}"
CONFIG_TYPE="${2:-default}"

# =============================================================================
# Environment Setup
# =============================================================================
echo "=============================================="
echo "16-GPU Multi-Node Training Launcher"
echo "=============================================="
echo "Master Node: $MASTER_ADDR"
echo "Worker Node: $WORKER_ADDR"
echo "Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "Exercise: $EXERCISE"
echo "Config: $CONFIG_TYPE"
echo "=============================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-finetune

# Set NCCL optimizations
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "Environment configured with NCCL optimizations"

# =============================================================================
# Exercise 1: LLM Training
# =============================================================================
if [ "$EXERCISE" == "exercise1" ]; then
    echo ""
    echo "Starting Exercise 1: LLM Fine-Tuning"
    echo "=============================================="

    # Select config file
    if [ "$CONFIG_TYPE" == "optimized" ]; then
        CONFIG_FILE="configs/training_config_optimized_noDS.yaml"
        echo "Using optimized configuration"
    else
        CONFIG_FILE="configs/training_config.yaml"
        echo "Using default configuration"
    fi

    SCRIPT_PATH="$REPO_PATH/exercise1/scripts/train_function_calling.py"
    WORK_DIR="$REPO_PATH/exercise1"

    # Create output directories
    mkdir -p /home/supreethlab/training/checkpoints
    mkdir -p /home/supreethlab/training/logs

    echo ""
    echo "Step 1: Starting worker node ($WORKER_ADDR)..."
    ssh $WORKER_ADDR "source ~/miniconda3/etc/profile.d/conda.sh && \
        conda activate llm-finetune && \
        export NCCL_IB_DISABLE=0 && \
        export NCCL_IB_GID_INDEX=3 && \
        export NCCL_NET_GDR_LEVEL=5 && \
        export NCCL_ALGO=Ring && \
        export NCCL_DEBUG=WARN && \
        cd $WORK_DIR && \
        torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
            --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            scripts/train_function_calling.py --config $CONFIG_FILE" &

    WORKER_PID=$!
    echo "Worker started with PID: $WORKER_PID"

    # Wait for worker to initialize
    sleep 5

    echo ""
    echo "Step 2: Starting master node ($MASTER_ADDR)..."
    echo ""

    cd $WORK_DIR
    torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        scripts/train_function_calling.py --config $CONFIG_FILE

    echo ""
    echo "=============================================="
    echo "Exercise 1 completed"
    echo "Checkpoints: /home/supreethlab/training/checkpoints/"
    echo "=============================================="

# =============================================================================
# Exercise 2: GPU Benchmarks
# =============================================================================
elif [ "$EXERCISE" == "exercise2" ]; then
    echo ""
    echo "Starting Exercise 2: GPU Benchmarks"
    echo "=============================================="

    SCRIPT_PATH="$REPO_PATH/exercise2/scripts/benchmark.py"
    WORK_DIR="$REPO_PATH/exercise2"
    OUTPUT_FILE="results/benchmark_16gpu_$(date +%Y%m%d_%H%M%S).json"

    echo ""
    echo "Step 1: Starting worker node ($WORKER_ADDR)..."
    ssh $WORKER_ADDR "source ~/miniconda3/etc/profile.d/conda.sh && \
        conda activate llm-finetune && \
        export NCCL_IB_DISABLE=0 && \
        export NCCL_IB_GID_INDEX=3 && \
        export NCCL_NET_GDR_LEVEL=5 && \
        export NCCL_ALGO=Ring && \
        export NCCL_DEBUG=WARN && \
        cd $WORK_DIR && \
        torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
            --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            scripts/benchmark.py --mode distributed" &

    WORKER_PID=$!
    echo "Worker started with PID: $WORKER_PID"

    # Wait for worker to initialize
    sleep 5

    echo ""
    echo "Step 2: Starting master node ($MASTER_ADDR)..."
    echo ""

    cd $WORK_DIR
    torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        scripts/benchmark.py --mode distributed --output $OUTPUT_FILE

    echo ""
    echo "=============================================="
    echo "Exercise 2 completed"
    echo "Results: $WORK_DIR/$OUTPUT_FILE"
    echo "=============================================="

else
    echo "Unknown exercise: $EXERCISE"
    echo "Usage: source multinode.sh [exercise1|exercise2] [optimized]"
fi

echo ""
echo "Done."
