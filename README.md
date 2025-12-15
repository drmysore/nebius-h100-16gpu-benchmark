# Nebius H100 16-GPU Multi-Node Benchmark

GPU cluster testing and LLM fine-tuning on a multi-node 16x NVIDIA H100 80GB configuration (2 nodes) on Nebius AI Cloud.

## Cluster Configuration

| Component | Details |
|-----------|---------|
| Nodes | 2 (poc-h100-node0, poc-node1) |
| GPUs per Node | 8x NVIDIA H100 80GB HBM3 |
| Total GPUs | 16 |
| Total GPU Memory | 1.27 TB (79.19 GB x 16) |
| Intra-node Interconnect | NVLink |
| Inter-node Interconnect | InfiniBand |
| CUDA | 12.1 |
| PyTorch | 2.3.0+cu121 |
| NCCL | 2.20.5 |

## Repository Structure

```
.
├── exercise1/                    # LLM Fine-Tuning for Function Calling
│   ├── scripts/
│   │   ├── train_function_calling.py
│   │   ├── preprocess_data.py
│   │   ├── evaluate.py
│   │   └── train_job.sbatch
│   ├── configs/
│   │   ├── training_config.yaml
│   │   └── ds_config_zero3.json
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       └── terraform.tfvars.example
│
├── exercise2/                    # GPU Cluster Acceptance Testing
│   ├── scripts/
│   │   └── benchmark.py
│   ├── configs/
│   │   └── benchmark_config.yaml
│   ├── results/
│   │   └── benchmark_16gpu.json
│   ├── tests/
│   │   └── test_benchmark.py
│   ├── k8s/
│   │   └── benchmark-job.yaml
│   ├── .github/workflows/
│   │   └── ci.yml
│   └── Dockerfile
│
└── README.md
```

## Exercise 1: LLM Fine-Tuning for Function Calling

Fine-tuning Qwen2-7B on the Glaive function calling dataset using LoRA with multi-node distributed training.

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2-7B-Instruct |
| Parameters | 7.7B total, 161M trainable (LoRA) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Precision | BF16 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 2 per GPU |
| Gradient Accumulation | 4 steps |
| Effective Batch Size | 128 (16 GPUs) |
| Learning Rate | 2e-5 |
| Scheduler | Cosine |
| Max Sequence Length | 4096 |

### Running Multi-Node Training

On Node 0 (master):

```bash
cd exercise1
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=10.2.0.129 --master_port=29500 \
  scripts/train_function_calling.py --config configs/training_config.yaml
```

On Node 1 (worker):

```bash
cd exercise1
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
  --master_addr=10.2.0.129 --master_port=29500 \
  scripts/train_function_calling.py --config configs/training_config.yaml
```

## Exercise 2: GPU Cluster Acceptance Testing

Benchmark results for validating multi-node GPU cluster performance.

### Benchmark Results Summary

| Test | Result | Expected | Status |
|------|--------|----------|--------|
| Multi-Node NCCL AllReduce | 434.99 GB/s | >350 GB/s | PASS |
| Distributed Training | 137,162 tokens/s | >500/GPU | PASS |
| World Size | 16 processes | 16 | PASS |

### NCCL AllReduce Bandwidth (16 GPUs across 2 nodes)

| Message Size | Bandwidth | Latency |
|--------------|-----------|---------|
| 1 MB | 22.30 GB/s | 0.088 ms |
| 10 MB | 124.92 GB/s | 0.157 ms |
| 100 MB | 304.03 GB/s | 0.647 ms |
| 1 GB | 434.99 GB/s | 4.520 ms |

### Distributed Training Benchmark

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 style transformer (12 layers, 768 hidden) |
| Batch Size | 8 per GPU (128 effective) |
| Sequence Length | 512 |
| World Size | 16 GPUs |
| Total Throughput | 137,162.0 tokens/second |
| Per-GPU Throughput | 8,572.6 tokens/second |

### Multi-Node Configuration

Node 0 (Master):

| Setting | Value |
|---------|-------|
| Hostname | computeinstance-e00e9ncccc9zh9nxw2 |
| Public IP | 89.169.98.13 |
| Private IP | 10.2.0.129 |
| Ranks | 0-7 |

Node 1 (Worker):

| Setting | Value |
|---------|-------|
| Hostname | computeinstance-e00f02hfdpb8fq4167 |
| Public IP | 89.169.98.157 |
| Private IP | 10.2.0.0 |
| Ranks | 8-15 |

### Running Multi-Node Benchmarks

On Node 0 (master):

```bash
cd exercise2
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=10.2.0.129 --master_port=29500 \
  scripts/benchmark.py --mode distributed
```

On Node 1 (worker):

```bash
cd exercise2
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
  --master_addr=10.2.0.129 --master_port=29500 \
  scripts/benchmark.py --mode distributed
```

## Performance Comparison: 8 GPUs vs 16 GPUs

| Metric | 8 GPUs (1 node) | 16 GPUs (2 nodes) |
|--------|-----------------|-------------------|
| NCCL Bandwidth | 442.21 GB/s | 434.99 GB/s |
| Bandwidth Efficiency | 100% (NVLink) | 98% (InfiniBand) |

The multi-node configuration maintains excellent interconnect bandwidth with only 2% overhead when scaling from NVLink to InfiniBand.

## Infrastructure

| Setting | Value |
|---------|-------|
| Cloud Provider | Nebius AI Cloud |
| Platform | gpu-h100-sxm |
| Preset | 8gpu-128vcpu-1600gb |
| GPU Cluster ID | computegpucluster-e00ge725pvdv7gyq28 |
| Inter-node Interconnect | InfiniBand |

## Benchmark Date

2025-12-15
