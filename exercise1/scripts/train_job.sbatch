#!/bin/bash
#SBATCH --job-name=llm-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1400G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/shared/logs/train_%j.log
#SBATCH --error=/mnt/shared/logs/train_%j.err
#SBATCH --exclusive

# =============================================================================
# Slurm Job Script for Multi-Node LLM Fine-tuning
# =============================================================================
# This script launches distributed training across 2 nodes with 8 H100 GPUs each
# Total: 16 GPUs
# =============================================================================

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "=============================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Load modules
module purge
module load cuda/12.4
module load nccl/2.21.5

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate llm-finetune

# Set paths
export PROJECT_DIR="/mnt/shared/llm-finetune"
export CONFIG_FILE="${PROJECT_DIR}/configs/training_config.yaml"
export DS_CONFIG="${PROJECT_DIR}/configs/ds_config_zero3.json"

# Create output directories
mkdir -p /mnt/shared/logs
mkdir -p /mnt/shared/checkpoints
mkdir -p /mnt/shared/logs/tensorboard

# =============================================================================
# NCCL and Network Configuration
# =============================================================================

# NCCL settings for InfiniBand
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# CUDA settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# PyTorch distributed settings
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# =============================================================================
# Distributed Training Configuration
# =============================================================================

# Get master node info
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "Master Node: $MASTER_ADDR:$MASTER_PORT"

# Calculate world size
NNODES=$SLURM_NNODES
NGPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * NGPUS_PER_NODE))

echo "World Size: $WORLD_SIZE (${NNODES} nodes x ${NGPUS_PER_NODE} GPUs)"

# =============================================================================
# HuggingFace Cache Configuration
# =============================================================================

export HF_HOME="/mnt/shared/.cache/huggingface"
export TRANSFORMERS_CACHE="/mnt/shared/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/mnt/shared/.cache/huggingface/datasets"
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# =============================================================================
# Memory and Performance Optimization
# =============================================================================

# Enable memory-efficient attention
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable tokenizers parallelism (conflicts with DataLoader workers)
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "Running pre-flight checks..."

# Check NVIDIA drivers
nvidia-smi || { echo "nvidia-smi failed"; exit 1; }

# Check NCCL
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"

# Check InfiniBand
ibstat | head -20 || echo "InfiniBand check skipped"

echo "Pre-flight checks completed"

# =============================================================================
# Launch Training
# =============================================================================

echo "Starting distributed training..."
echo "Timestamp: $(date)"

# Using DeepSpeed launcher with Slurm integration
srun --kill-on-bad-exit=1 \
    deepspeed \
    --hostfile /dev/null \
    --num_nodes=${NNODES} \
    --num_gpus=${NGPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${PROJECT_DIR}/scripts/train_function_calling.py \
    --config ${CONFIG_FILE}

# Capture exit code
EXIT_CODE=$?

echo "=============================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End Timestamp: $(date)"
echo "=============================================="

# =============================================================================
# Post-training Tasks
# =============================================================================

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training successful!"
    
    # Copy final checkpoint to persistent storage
    if [ -d "/mnt/shared/checkpoints/final" ]; then
        echo "Final model saved at: /mnt/shared/checkpoints/final"
    fi
    
    # Generate training summary
    echo "Training Summary:" >> /mnt/shared/logs/train_${SLURM_JOB_ID}_summary.txt
    echo "Job ID: $SLURM_JOB_ID" >> /mnt/shared/logs/train_${SLURM_JOB_ID}_summary.txt
    echo "Nodes: $SLURM_NNODES" >> /mnt/shared/logs/train_${SLURM_JOB_ID}_summary.txt
    echo "GPUs: $WORLD_SIZE" >> /mnt/shared/logs/train_${SLURM_JOB_ID}_summary.txt
    echo "Duration: $SECONDS seconds" >> /mnt/shared/logs/train_${SLURM_JOB_ID}_summary.txt
else
    echo "Training failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
