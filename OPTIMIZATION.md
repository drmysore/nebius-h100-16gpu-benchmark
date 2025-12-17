# GPU Cluster Performance Optimization Guide

This document details the performance analysis, identified bottlenecks, and optimizations applied to the Nebius H100 GPU cluster for LLM fine-tuning workloads.

## Table of Contents

1. [Hardware Configuration](#hardware-configuration)
2. [Performance Baseline (Before)](#performance-baseline-before)
3. [Identified Bottlenecks](#identified-bottlenecks)
4. [Optimizations Applied](#optimizations-applied)
5. [Configuration Changes](#configuration-changes)
6. [Performance Results (After)](#performance-results-after)
7. [Monitoring Commands](#monitoring-commands)

---

## Hardware Configuration

### Cluster Specifications

| Component | Specification |
|-----------|---------------|
| Platform | Nebius AI Cloud |
| Nodes | 2 |
| GPUs per Node | 8x NVIDIA H100 80GB HBM3 |
| Total GPUs | 16 |
| GPU Memory | 80 GB HBM3 per GPU |
| Intra-node Interconnect | NVLink 4.0 (18 links, 900 GB/s bidirectional) |
| Inter-node Interconnect | InfiniBand NDR (8x mlx5 NICs) |
| NVLink Bandwidth | 26.562 GB/s per link x 18 = 478 GB/s |
| CUDA Version | 12.1 |
| PyTorch Version | 2.3.0+cu121 |
| NCCL Version | 2.20.5 |

### GPU Topology

```
All 8 GPUs per node connected via NV18 (18 NVLinks each)
Node 0: GPUs 0-7, NUMA Node 0-1, CPU cores 0-127
Node 1: GPUs 0-7, NUMA Node 0-1, CPU cores 0-127
Inter-node: InfiniBand (8x Mellanox mlx5)
```

---

## Performance Baseline (Before)

### Benchmark Results - Before Optimization

| Metric | 8 GPUs (1 Node) | 16 GPUs (2 Nodes) |
|--------|-----------------|-------------------|
| NCCL AllReduce Bandwidth | 442.28 GB/s | 435.91 GB/s |
| Total Training Throughput | 331,842 tokens/s | 136,970 tokens/s |
| Per-GPU Throughput | 41,480 tokens/s | 8,561 tokens/s |
| Scaling Efficiency | 100% (baseline) | 21% |

### Original Configuration

```yaml
# training_config.yaml - BEFORE optimization

training:
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4
  # Effective batch size = 2 * 4 * 16 GPUs = 128

  learning_rate: 2.0e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03

  optim: "adamw_torch_fused"
  bf16: true
  tf32: true

  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false

  dataloader_num_workers: 4
  dataloader_pin_memory: true

distributed:
  deepspeed: null  # DeepSpeed disabled
  ddp_backend: "nccl"
```

### Profiling Data - Before

From Nsight Systems profiling:

| Metric | Value |
|--------|-------|
| GEMM Kernel Time | 98% of total |
| SM Utilization | 0-100% (fluctuating) |
| Memory Controller Utilization | 0-69% |
| GPU Memory Used | 60-80 GB per GPU |
| Power Draw | 155-630 W per GPU |

---

## Identified Bottlenecks

### 1. Multi-Node Communication Overhead

**Problem:** Per-GPU throughput drops 80% when scaling from 1 node to 2 nodes.

| Evidence | Value |
|----------|-------|
| 8-GPU per-GPU throughput | 41,480 tokens/s |
| 16-GPU per-GPU throughput | 8,561 tokens/s |
| Efficiency loss | 79% |

**Root Cause:**
- InfiniBand latency (~1-2 us) vs NVLink latency (~100 ns)
- Gradient synchronization blocking compute
- Small batch size not hiding communication latency
- DDP (DistributedDataParallel) synchronous updates

### 2. GPU Memory Underutilization

**Problem:** Only using 75-98% of available GPU memory.

| Metric | Current | Available |
|--------|---------|-----------|
| Memory per GPU | 60-80 GB | 80 GB |
| Batch size | 2 | Could be 4-8 |

**Root Cause:**
- Conservative batch size setting
- Gradient checkpointing reducing memory but also throughput

### 3. Suboptimal Data Loading

**Problem:** Data loading may bottleneck GPU compute.

| Metric | Current | Optimal |
|--------|---------|---------|
| DataLoader workers | 4 | 8-16 |
| Prefetch factor | Default (2) | 4 |

### 4. Missing DeepSpeed Optimizations

**Problem:** Using basic DDP instead of ZeRO optimization.

| Feature | DDP | DeepSpeed ZeRO-3 |
|---------|-----|------------------|
| Memory efficiency | Low | High |
| Communication overlap | No | Yes |
| Gradient partitioning | No | Yes |
| Optimizer state sharding | No | Yes |

### 5. NCCL Configuration Not Tuned

**Problem:** Default NCCL settings not optimized for InfiniBand.

Missing optimizations:
- GPU Direct RDMA not explicitly enabled
- Suboptimal buffer sizes
- No algorithm tuning for message sizes

---

## Optimizations Applied

### Optimization 1: Increase Batch Size

**Rationale:** Fill GPU memory to maximize compute utilization and amortize communication overhead.

| Parameter | Before | After |
|-----------|--------|-------|
| per_device_train_batch_size | 2 | 4 |
| gradient_accumulation_steps | 4 | 2 |
| Effective batch size | 128 | 128 |

**Expected Impact:** 30-50% throughput increase

### Optimization 2: Enable DeepSpeed ZeRO-3

**Rationale:** Overlap communication with compute, shard optimizer states across GPUs.

| Feature | Benefit |
|---------|---------|
| overlap_comm | Hides communication latency |
| contiguous_gradients | Faster gradient reduction |
| reduce_bucket_size | Optimized for H100 memory |
| stage3_prefetch | Prefetch parameters before needed |

**Expected Impact:** 100-200% throughput increase for multi-node

### Optimization 3: NCCL Environment Tuning

**Rationale:** Enable GPU Direct RDMA and tune for InfiniBand.

| Variable | Value | Purpose |
|----------|-------|---------|
| NCCL_IB_DISABLE | 0 | Enable InfiniBand |
| NCCL_IB_GID_INDEX | 3 | RoCE v2 GID |
| NCCL_NET_GDR_LEVEL | 5 | GPU Direct RDMA |
| NCCL_IB_QPS_PER_CONNECTION | 4 | More queue pairs |
| NCCL_ALGO | Ring | Optimal for AllReduce |

**Expected Impact:** 10-20% communication speedup

### Optimization 4: Increase DataLoader Workers

**Rationale:** Prevent CPU data loading from bottlenecking GPU compute.

| Parameter | Before | After |
|-----------|--------|-------|
| dataloader_num_workers | 4 | 8 |
| dataloader_prefetch_factor | 2 | 4 |

**Expected Impact:** 5-10% throughput increase

### Optimization 5: Torch Compile (Optional)

**Rationale:** JIT compile model for optimized kernel execution.

| Parameter | Before | After |
|-----------|--------|-------|
| torch_compile | false | true |
| torch_compile_backend | N/A | inductor |

**Expected Impact:** 10-30% kernel speedup

---

## Configuration Changes

### BEFORE: training_config.yaml

```yaml
training:
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4

  learning_rate: 2.0e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03

  optim: "adamw_torch_fused"
  weight_decay: 0.01
  max_grad_norm: 1.0

  bf16: true
  fp16: false
  tf32: true

  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false

  logging_steps: 10
  save_strategy: "steps"
  save_steps: 50
  eval_strategy: "steps"
  eval_steps: 50

  dataloader_num_workers: 4
  dataloader_pin_memory: true

distributed:
  deepspeed: null
  ddp_backend: "nccl"
```

### AFTER: training_config_optimized.yaml

```yaml
training:
  # Increased batch size to fill GPU memory
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 2
  # Effective batch size = 4 * 2 * 16 GPUs = 128 (same total)

  learning_rate: 2.0e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03

  optim: "adamw_torch_fused"
  weight_decay: 0.01
  max_grad_norm: 1.0

  bf16: true
  fp16: false
  tf32: true

  # Keep gradient checkpointing for memory efficiency
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false

  logging_steps: 10
  save_strategy: "steps"
  save_steps: 50
  eval_strategy: "steps"
  eval_steps: 50

  # Increased data loading parallelism
  dataloader_num_workers: 8
  dataloader_pin_memory: true
  dataloader_prefetch_factor: 4

distributed:
  # Enable DeepSpeed ZeRO-3 for multi-node efficiency
  deepspeed: "configs/ds_config_zero3_optimized.json"
  ddp_backend: "nccl"
```

### BEFORE: ds_config_zero3.json

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3
  },
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

### AFTER: ds_config_zero3_optimized.json

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 500000000,
    "stage3_param_persistence_threshold": 1000000,
    "sub_group_size": 1000000000,
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000000000,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "wall_clock_breakdown": false,
  "communication_data_type": "bf16"
}
```

### BEFORE: Environment Variables

```bash
# No specific NCCL tuning
```

### AFTER: optimize_env.sh

```bash
#!/bin/bash
# NCCL Optimizations for InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=136
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

# CUDA Memory Optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Reduce Debug Overhead
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=WARN

# OMP Settings
export OMP_NUM_THREADS=8
```

---

## Performance Results (After)

### Expected Improvements

| Optimization | Impact |
|--------------|--------|
| Increased batch size (2 to 4) | +30-50% |
| DeepSpeed ZeRO-3 with overlap | +100-200% |
| NCCL tuning | +10-20% |
| DataLoader optimization | +5-10% |
| **Combined** | **3-5x improvement** |

### Projected Results

| Metric | Before | After (Projected) |
|--------|--------|-------------------|
| 16-GPU Total Throughput | 136,970 tokens/s | 400,000-600,000 tokens/s |
| 16-GPU Per-GPU Throughput | 8,561 tokens/s | 25,000-37,500 tokens/s |
| Scaling Efficiency (vs 8-GPU) | 21% | 60-90% |
| GPU Memory Utilization | 75% | 90-95% |
| Communication Overhead | High | Low (overlapped) |

---

## Monitoring Commands

### Real-Time GPU Monitoring

```bash
# GPU utilization, memory, temperature, power
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw --format=csv'

# Detailed per-second metrics
nvidia-smi dmon -s pucvmet -d 1

# GPU topology and NVLink status
nvidia-smi topo -m
nvidia-smi nvlink -s
```

### Profiling Commands

```bash
# Nsight Systems - System-wide profiling
nsys profile --stats=true --output=training_profile \
    torchrun --nproc_per_node=8 scripts/train_function_calling.py

# View profile report
nsys stats training_profile.nsys-rep

# NCCL Debug (for troubleshooting)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### Training Metrics

```bash
# Monitor training logs
tail -f /home/supreethlab/training/logs/training.log

# TensorBoard
tensorboard --logdir=/home/supreethlab/training/logs --port=6006

# Check GPU memory during training
watch -n 5 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

---

## Running Optimized Training

### Single Node (8 GPUs)

```bash
source optimize_env.sh
cd /home/supreethlab/repos/nebius-h100-16gpu-benchmark/exercise1

torchrun --nproc_per_node=8 \
    scripts/train_function_calling.py \
    --config configs/training_config_optimized.yaml
```

### Multi-Node (16 GPUs)

```bash
# On both nodes, source environment first
source optimize_env.sh

# Node 0 (Master)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=10.2.0.129 --master_port=29500 \
    scripts/train_function_calling.py \
    --config configs/training_config_optimized.yaml

# Node 1 (Worker)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=10.2.0.129 --master_port=29500 \
    scripts/train_function_calling.py \
    --config configs/training_config_optimized.yaml
```

---

## Summary

| Area | Before | After | Change |
|------|--------|-------|--------|
| Batch Size | 2 | 4 | +100% |
| DataLoader Workers | 4 | 8 | +100% |
| DeepSpeed | Disabled | ZeRO-3 | Enabled |
| Communication Overlap | No | Yes | Enabled |
| NCCL Tuning | Default | Optimized | Tuned |
| GPU Memory Usage | 75% | 90-95% | +20% |
| Expected Throughput | 1x | 3-5x | +200-400% |

---

## References

- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)
- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NVIDIA H100 Specifications](https://www.nvidia.com/en-us/data-center/h100/)

---

## Author

Supreeth Mysore

## Date

2025-12-16
