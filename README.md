# Nebius H100 16-GPU Multi-Node Benchmark Results

This repository contains GPU cluster acceptance testing results for a multi-node configuration with 16x NVIDIA H100 80GB GPUs (2 nodes) on Nebius AI Cloud.

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

## Benchmark Results Summary

| Test | Result | Expected | Status |
|------|--------|----------|--------|
| Multi-Node NCCL AllReduce | 434.99 GB/s | >350 GB/s | PASS |
| Distributed Training | 137,162 tokens/s | >500/GPU | PASS |
| World Size | 16 processes | 16 | PASS |

## Detailed Results

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

## Multi-Node Configuration

### Node 0 (Master)

| Setting | Value |
|---------|-------|
| Hostname | computeinstance-e00e9ncccc9zh9nxw2 |
| Public IP | 89.169.98.13 |
| Private IP | 10.2.0.129 |
| Ranks | 0-7 |

### Node 1 (Worker)

| Setting | Value |
|---------|-------|
| Hostname | computeinstance-e00f02hfdpb8fq4167 |
| Public IP | 89.169.98.157 |
| Private IP | 10.2.0.0 |
| Ranks | 8-15 |

## Running the Multi-Node Benchmark

On Node 0 (master):

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=10.2.0.129 --master_port=29500 \
  scripts/benchmark.py --mode distributed
```

On Node 1 (worker), run simultaneously:

```bash
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

## Repository Contents

| File | Description |
|------|-------------|
| scripts/benchmark.py | Main benchmark script |
| configs/benchmark_config.yaml | Benchmark configuration |
| results/benchmark_16gpu.json | Full benchmark results |

## Infrastructure

| Setting | Value |
|---------|-------|
| Platform | gpu-h100-sxm |
| Preset | 8gpu-128vcpu-1600gb |
| GPU Cluster ID | computegpucluster-e00ge725pvdv7gyq28 |
| Inter-node Interconnect | InfiniBand |

## Benchmark Date

2025-12-15
