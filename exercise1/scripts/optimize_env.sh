#!/bin/bash
# optimize_env.sh - Environment variables for optimized multi-node training
# Source this script before running training: source scripts/optimize_env.sh

# =============================================================================
# NCCL Optimizations for InfiniBand
# =============================================================================
export NCCL_IB_DISABLE=0                    # Enable InfiniBand
export NCCL_IB_GID_INDEX=3                  # RoCE v2 GID index
export NCCL_NET_GDR_LEVEL=5                 # GPU Direct RDMA level
export NCCL_IB_QPS_PER_CONNECTION=4         # Queue pairs per connection
export NCCL_IB_TC=136                       # Traffic class
export NCCL_ALGO=Ring                       # AllReduce algorithm
export NCCL_PROTO=Simple                    # Protocol

# =============================================================================
# CUDA Memory Optimizations
# =============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1        # Limit CUDA connections
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# Reduce Debug Overhead in Production
# =============================================================================
export TORCH_DISTRIBUTED_DEBUG=OFF          # Disable distributed debug
export NCCL_DEBUG=WARN                      # Only show warnings

# =============================================================================
# CPU Thread Settings
# =============================================================================
export OMP_NUM_THREADS=8                    # OpenMP threads
export MKL_NUM_THREADS=8                    # MKL threads

# =============================================================================
# HuggingFace Cache (shared across nodes)
# =============================================================================
export HF_HOME="/home/supreethlab/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/supreethlab/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/home/supreethlab/.cache/huggingface/datasets"

# =============================================================================
# Tokenizers
# =============================================================================
export TOKENIZERS_PARALLELISM=false         # Avoid conflicts with DataLoader

echo "Environment optimized for multi-node training"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL"
echo "NCCL_ALGO=$NCCL_ALGO"
