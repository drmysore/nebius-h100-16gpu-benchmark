# Nebius H100 GPU Cluster Demo Guide

This document provides a complete walkthrough for demonstrating the GPU cluster deployment, validation, and LLM fine-tuning capabilities on Nebius AI Cloud.

## Table of Contents

1. [Overview](#overview)
2. [Cluster Architecture](#cluster-architecture)
3. [Pre-Demo Checklist](#pre-demo-checklist)
4. [Demo Part 1: Hardware Validation](#demo-part-1-hardware-validation)
5. [Demo Part 2: GPU Benchmarks (Exercise 2)](#demo-part-2-gpu-benchmarks-exercise-2)
6. [Demo Part 3: LLM Fine-Tuning (Exercise 1)](#demo-part-3-llm-fine-tuning-exercise-1)
7. [Demo Part 4: Performance Analysis](#demo-part-4-performance-analysis)
8. [Results Summary](#results-summary)
9. [Troubleshooting](#troubleshooting)
10. [Appendix](#appendix)

---

## Overview

This demonstration showcases two primary capabilities:

1. **Exercise 1 - LLM Fine-Tuning**: Training a 7B parameter language model for function calling using distributed training across multiple GPUs and nodes.

2. **Exercise 2 - GPU Cluster Acceptance Testing**: Validating cluster performance through comprehensive benchmarks including compute, memory bandwidth, and inter-GPU communication.

### Repository Structure

```
nebius-h100-16gpu-benchmark/
├── exercise1/                    # LLM Fine-Tuning
│   ├── scripts/
│   │   ├── train_function_calling.py
│   │   └── train_job.sbatch
│   └── configs/
│       ├── training_config.yaml
│       └── training_config_optimized_noDS.yaml
├── exercise2/                    # GPU Benchmarks
│   ├── scripts/
│   │   └── benchmark.py
│   └── results/
├── multinode.sh                  # Multi-node launcher
├── hardware_info.sh              # Hardware discovery
├── OPTIMIZATION.md               # Performance tuning guide
└── DEMO.md                       # This file
```

---

## Cluster Architecture

### Hardware Specifications

| Component | Per Node | Total (2 Nodes) |
|-----------|----------|-----------------|
| CPU | Intel Xeon Platinum 8468 (128 cores) | 256 cores |
| RAM | 1.5 TB DDR5 | 3 TB |
| GPU | 8x NVIDIA H100 80GB HBM3 | 16 GPUs |
| GPU Memory | 640 GB HBM3 | 1.28 TB |
| InfiniBand | 8x Mellanox ConnectX (400 Gb/s) | 16 ports |

### Network Topology

| Connection Type | Technology | Bandwidth |
|-----------------|------------|-----------|
| GPU-to-GPU (intra-node) | NVLink 4.0 | 900 GB/s bidirectional |
| Node-to-Node (inter-node) | InfiniBand NDR | 400 Gb/s per port |

### Node Configuration

| Node | Hostname | IP Address | Role |
|------|----------|------------|------|
| 0 | computeinstance-e00e9ncccc9zh9nxw2 | 10.2.0.129 | Master |
| 1 | computeinstance-e00f02hfdpb8fq4167 | 10.2.0.0 | Worker |

---

## Pre-Demo Checklist

### 1. Verify SSH Connectivity

```bash
# From master node, verify worker is reachable
ssh 10.2.0.0 "hostname && nvidia-smi -L | head -2"
```

### 2. Verify Conda Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-finetune
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 3. Verify GPU Availability

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

### 4. Sync Repository to Worker Node

```bash
rsync -avz /home/supreethlab/repos/nebius-h100-16gpu-benchmark/ \
    10.2.0.0:/home/supreethlab/repos/nebius-h100-16gpu-benchmark/
```

---

## Demo Part 1: Hardware Validation

### Show Cluster Hardware Configuration

```bash
cd /home/supreethlab/repos/nebius-h100-16gpu-benchmark
./hardware_info.sh summary
```

Expected output:
```
==============================================
CLUSTER SUMMARY
==============================================
Node 0 (Master): computeinstance-e00e9ncccc9zh9nxw2 - 10.2.0.129
Node 1 (Worker): computeinstance-e00f02hfdpb8fq4167 - 10.2.0.0

Per Node:
  - CPU: Intel Xeon Platinum 8468 (128 cores)
  - RAM: 1.5 TB
  - GPUs: 8x NVIDIA H100 80GB HBM3
  - InfiniBand: 8x Mellanox ConnectX (400 Gb/s each)

Total Cluster:
  - Nodes: 2
  - Total GPUs: 16
  - Total GPU Memory: 1.28 TB
  - Total RAM: 3 TB
  - Interconnect: NVLink (intra-node), InfiniBand NDR (inter-node)
```

### Show GPU Topology

```bash
./hardware_info.sh gpu
```

Key points to highlight:
- All 8 GPUs connected via NV18 (18 NVLink connections each)
- GPUs 0-3 on NUMA node 0, GPUs 4-7 on NUMA node 1
- 8 InfiniBand NICs for inter-node communication

### Show InfiniBand Status

```bash
./hardware_info.sh ib
```

Key points:
- 8 Mellanox ConnectX adapters per node
- All ports active at 400 Gb/s
- InfiniBand link layer (not Ethernet)

---

## Demo Part 2: GPU Benchmarks (Exercise 2)

### Option A: Single Node Benchmarks (8 GPUs)

```bash
cd /home/supreethlab/repos/nebius-h100-8gpu-benchmark
source singlenode.sh exercise2
```

### Option B: Multi-Node Benchmarks (16 GPUs)

```bash
cd /home/supreethlab/repos/nebius-h100-16gpu-benchmark
source multinode.sh exercise2
```

### Benchmark Results Explanation

#### GPU Health Check

Verifies all GPUs are visible and have expected memory.

| Metric | Expected | Description |
|--------|----------|-------------|
| GPU Count | 8 (or 16) | All GPUs detected |
| Memory | 80 GB each | HBM3 memory available |
| Status | PASS | No hardware errors |

#### Single GPU Benchmarks

| Test | Metric | 8-GPU Result | Description |
|------|--------|--------------|-------------|
| MatMul (BF16) | TFLOPS | 730.97 | Matrix multiplication performance |
| Memory Bandwidth | TB/s | 3.02 | HBM3 memory throughput |

#### Distributed Benchmarks

| Test | Metric | 8-GPU | 16-GPU | Description |
|------|--------|-------|--------|-------------|
| NCCL AllReduce | GB/s | 442.28 | 435.91 | Collective communication bandwidth |
| Training Throughput | tokens/s | 331,842 | 136,970 | Simulated training performance |

### View Benchmark Results

```bash
# View results file
cat exercise2/results/benchmark_16gpu.json | python -m json.tool
```

---

## Demo Part 3: LLM Fine-Tuning (Exercise 1)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2-7B-Instruct |
| Method | LoRA (Low-Rank Adaptation) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Trainable Parameters | 161M (2.1% of 7.7B) |
| Precision | BF16 |
| Batch Size | 2 per GPU |
| Gradient Accumulation | 4 steps |
| Effective Batch Size | 128 (16 GPUs) |
| Max Sequence Length | 4096 |
| Training Steps | 100 (demo) |

### Option A: Single Node Training (8 GPUs)

```bash
cd /home/supreethlab/repos/nebius-h100-8gpu-benchmark
source singlenode.sh exercise1
```

### Option B: Multi-Node Training (16 GPUs)

```bash
cd /home/supreethlab/repos/nebius-h100-16gpu-benchmark
source multinode.sh exercise1
```

### Option C: Optimized Configuration

```bash
# Single node optimized
source singlenode.sh exercise1 optimized

# Multi-node optimized
source multinode.sh exercise1 optimized
```

### Monitor Training Progress

Training logs show:
- Loss decreasing from ~1.7 to ~0.4 over 100 steps
- Evaluation at steps 50 and 100
- Checkpoints saved at steps 50 and 100

Expected output pattern:
```
Step 10:  loss=1.2292, lr=1.97e-05
Step 20:  loss=0.6736, lr=1.85e-05
Step 30:  loss=0.5045, lr=1.64e-05
...
Step 100: loss=0.3918, lr=1.31e-07 [Eval: 0.3958]
```

### Training Success Criteria

| Criterion | Expected | How to Verify |
|-----------|----------|---------------|
| Completion | Exit code 0 | Training finishes without error |
| Final Loss | < 0.5 | Check training logs |
| Checkpoint | Saved | `ls /home/supreethlab/training/checkpoints/final/` |
| No OOM | None | No CUDA out of memory errors |

### View Training Output

```bash
# List checkpoints
ls -la /home/supreethlab/training/checkpoints/

# Check final model
ls -la /home/supreethlab/training/checkpoints/final/
```

---

## Demo Part 4: Performance Analysis

### GPU Utilization During Training

```bash
# Real-time monitoring
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv'

# Detailed metrics
nvidia-smi dmon -s pucvmet -d 1
```

### Key Performance Metrics

| Metric | 8 GPUs | 16 GPUs | Notes |
|--------|--------|---------|-------|
| Training Loss | 0.3895 | 0.3918 | Similar convergence |
| Eval Loss | 0.396 | 0.3958 | Similar quality |
| NCCL Bandwidth | 442 GB/s | 436 GB/s | 98% scaling efficiency |
| Time per Step | ~1.7s | ~1.9s | Communication overhead |

### Scaling Analysis

| Configuration | GPUs | Interconnect | Per-GPU Throughput |
|---------------|------|--------------|-------------------|
| Single Node | 8 | NVLink | 41,480 tokens/s |
| Multi-Node | 16 | InfiniBand | 8,561 tokens/s |

The per-GPU throughput reduction in multi-node configuration is due to:
- InfiniBand latency vs NVLink
- Gradient synchronization overhead across nodes
- Communication not fully overlapped with compute

### Optimization Impact

| Optimization | Impact |
|--------------|--------|
| Increased batch size (2 to 4) | +30-50% throughput |
| More DataLoader workers (4 to 8) | +5-10% throughput |
| NCCL tuning (GPU Direct RDMA) | +10-20% communication |
| DeepSpeed ZeRO-3 | +100-200% (requires script modification) |

---

## Results Summary

### Exercise 2: Benchmark Results

| Test | 8 GPUs | 16 GPUs | Status |
|------|--------|---------|--------|
| GPU Health | 8/8 PASS | 16/16 PASS | PASS |
| MatMul (BF16) | 730.97 TFLOPS | N/A | PASS |
| Memory Bandwidth | 3.02 TB/s | N/A | PASS |
| NCCL AllReduce | 442.28 GB/s | 435.91 GB/s | PASS |
| Training Throughput | 331,842 tok/s | 136,970 tok/s | PASS |

### Exercise 1: Training Results

| Metric | 8 GPUs | 16 GPUs |
|--------|--------|---------|
| Model | Qwen/Qwen2-7B-Instruct | Qwen/Qwen2-7B-Instruct |
| Training Steps | 100 | 100 |
| Final Training Loss | 0.3895 | 0.3918 |
| Final Eval Loss | 0.396 | 0.3958 |
| Checkpoint Location | `/home/supreethlab/training/checkpoints/final/` | `/home/supreethlab/training/checkpoints_16gpu/final/` |

---

## Troubleshooting

### Common Issues and Solutions

#### NCCL Timeout (Multi-Node)

```
torch.distributed.DistStoreError: Timed out after 901 seconds waiting for clients
```

**Cause:** Worker node not started or not reachable.

**Solution:**
1. Verify SSH connectivity: `ssh 10.2.0.0 hostname`
2. Start worker node first, then master
3. Use `multinode.sh` which handles ordering automatically

#### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Cause:** Batch size too large for GPU memory.

**Solution:**
1. Reduce `per_device_train_batch_size` in config
2. Enable gradient checkpointing (already enabled by default)
3. Use non-optimized config: `source multinode.sh exercise1`

#### InfiniBand Warnings

```
libibverbs: Warning: couldn't load driver 'libvmw_pvrdma-rdmav34.so'
```

**Cause:** Harmless warning about VMware driver not found.

**Solution:** Ignore - does not affect functionality.

#### DeepSpeed ZeRO-3 Error

```
ValueError: Model was not initialized with Zero-3
```

**Cause:** DeepSpeed ZeRO-3 requires model initialization after TrainingArguments.

**Solution:** Use `training_config_optimized_noDS.yaml` which disables DeepSpeed.

### Verification Commands

```bash
# Check GPU status
nvidia-smi

# Check InfiniBand
ibstat | grep -E "State|Rate"

# Check node connectivity
ping -c 3 10.2.0.0

# Check disk space
df -h /home/supreethlab/training/
```

---

## Appendix

### A. File Locations

| Item | Path |
|------|------|
| 16-GPU Repository | `/home/supreethlab/repos/nebius-h100-16gpu-benchmark/` |
| 8-GPU Repository | `/home/supreethlab/repos/nebius-h100-8gpu-benchmark/` |
| Training Checkpoints | `/home/supreethlab/training/checkpoints/` |
| Training Logs | `/home/supreethlab/training/logs/` |
| Benchmark Results | `exercise2/results/` |

### B. Configuration Files

| File | Purpose |
|------|---------|
| `training_config.yaml` | Default training configuration |
| `training_config_optimized_noDS.yaml` | Optimized (no DeepSpeed) |
| `ds_config_zero3.json` | DeepSpeed ZeRO-3 settings |
| `benchmark_config.yaml` | Benchmark thresholds |

### C. Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `multinode.sh` | 16-GPU launcher | `source multinode.sh [exercise] [config]` |
| `singlenode.sh` | 8-GPU launcher | `source singlenode.sh [exercise] [config]` |
| `hardware_info.sh` | Hardware discovery | `./hardware_info.sh [option]` |

### D. Environment Variables

```bash
# NCCL Optimizations
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_ALGO=Ring

# CUDA Settings
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
```

### E. Useful Commands

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# View GPU topology
nvidia-smi topo -m

# Check InfiniBand status
ibstat

# View training logs
tail -f /home/supreethlab/training/logs/*.log

# Kill stuck processes
pkill -f torchrun
```

---

## Demo Timeline (Suggested)

| Time | Section | Duration |
|------|---------|----------|
| 0:00 | Overview and architecture | 2 min |
| 2:00 | Hardware validation | 2 min |
| 4:00 | Exercise 2: Benchmarks | 3 min |
| 7:00 | Exercise 1: Training demo | 5 min |
| 12:00 | Results and analysis | 3 min |
| 15:00 | Q&A | 5 min |

**Total Duration: 15-20 minutes**

---

## Author

Supreeth Mysore

## Date

2025-12-17

## Version

1.0
