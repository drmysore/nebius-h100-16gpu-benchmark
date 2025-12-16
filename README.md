# Nebius H100 GPU Cluster - Multi-Node LLM Fine-Tuning and Acceptance Testing

This repository contains solutions for two exercises demonstrating multi-node GPU cluster deployment, validation, and ML workload execution on Nebius AI Cloud.

- **Exercise 1**: Multi-node LLM fine-tuning for function calling using Slurm/Soperator
- **Exercise 2**: GPU cluster acceptance testing with automated benchmarks and CI/CD

## Cluster Configuration

| Component | Specification |
|-----------|---------------|
| Platform | Nebius AI Cloud |
| Nodes | 2 |
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
│   │   └── train_job.sbatch      # Slurm job script
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

---

## Exercise 1: LLM Fine-Tuning for Function Calling

Fine-tuning Qwen2-7B on the Glaive function calling dataset using LoRA and DeepSpeed ZeRO-3 across 2 nodes.

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

### Running Training via Slurm (Recommended)

The primary way to run multi-node training is via Slurm/Soperator:

```bash
cd exercise1

# Submit the training job
sbatch scripts/train_job.sbatch

# Monitor job status
squeue -u $USER
watch -n 5 squeue -u $USER

# View training logs
tail -f /mnt/shared/logs/train_<JOB_ID>.log

# Check GPU utilization across nodes
srun --jobid=<JOB_ID> nvidia-smi -l 5
```

### Alternative: Manual torchrun (Debug Mode)

For debugging or manual execution across nodes:

```bash
# On Node 0 (master)
cd exercise1
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<MASTER_PRIVATE_IP> --master_port=29500 \
    scripts/train_function_calling.py --config configs/training_config.yaml

# On Node 1 (worker)
cd exercise1
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=<MASTER_PRIVATE_IP> --master_port=29500 \
    scripts/train_function_calling.py --config configs/training_config.yaml
```

### Success Criteria

Training is considered successful when:

| Criterion | Expected | Location |
|-----------|----------|----------|
| Job completes | Exit code 0 | `squeue` shows COMPLETED |
| Final checkpoint saved | `final/` directory exists | `/mnt/shared/checkpoints/final/` |
| Training loss converges | < 0.5 | Training logs |
| No OOM errors | None | Error logs |
| Throughput | > 2000 tokens/s/GPU | Training logs |

Output artifacts:
- **Checkpoints**: `/mnt/shared/checkpoints/`
- **Logs**: `/mnt/shared/logs/train_<JOB_ID>.log`
- **TensorBoard**: `/mnt/shared/logs/tensorboard/`
- **Summary**: `/mnt/shared/logs/train_<JOB_ID>_summary.txt`

---

## Exercise 2: GPU Cluster Acceptance Testing

Comprehensive benchmark suite for validating multi-node GPU cluster performance.

### Acceptance Test Results Summary

| Test | Metric | Threshold | Observed | Status |
|------|--------|-----------|----------|--------|
| Multi-Node Connectivity | All nodes reachable | 2/2 | 2/2 | PASS |
| NCCL AllReduce (InfiniBand) | GB/s | > 350 | 434.99 | PASS |
| Training Throughput | tokens/s | > 500/GPU | 8,573/GPU | PASS |
| Scaling Efficiency | % | > 90% | 98% | PASS |

**Overall Status: PASS**

### Detailed Benchmark Results

#### NCCL AllReduce Bandwidth (16 GPUs across 2 nodes via InfiniBand)

| Message Size | Bandwidth | Latency |
|--------------|-----------|---------|
| 1 MB | 22.30 GB/s | 0.088 ms |
| 10 MB | 124.92 GB/s | 0.157 ms |
| 100 MB | 304.03 GB/s | 0.647 ms |
| 1 GB | 434.99 GB/s | 4.520 ms |

#### Distributed Training Throughput

| Metric | Value |
|--------|-------|
| Total throughput | 137,162 tokens/s |
| Per-GPU throughput | 8,573 tokens/s |
| World size | 16 GPUs (2 nodes) |
| Batch size | 8 per GPU (128 effective) |
| Sequence length | 512 |

### Multi-Node Configuration

| Node | Role | Ranks |
|------|------|-------|
| gpu-node-0 | Master | 0-7 |
| gpu-node-1 | Worker | 8-15 |

### Performance Comparison: 8 GPUs vs 16 GPUs

| Metric | 8 GPUs (1 node) | 16 GPUs (2 nodes) | Efficiency |
|--------|-----------------|-------------------|------------|
| NCCL Bandwidth | 442.21 GB/s | 434.99 GB/s | 98% |
| Interconnect | NVLink | InfiniBand | - |

The multi-node configuration maintains excellent interconnect bandwidth with only 2% overhead when scaling from NVLink to InfiniBand.

### Running Benchmarks

#### Using Docker (Recommended)

```bash
# Pull the container image
docker pull ghcr.io/<your-org>/nebius-h100-16gpu-benchmark:latest

# Multi-node distributed benchmarks require torchrun across nodes
# On each node:
docker run --gpus all --network host \
    -e MASTER_ADDR=<MASTER_PRIVATE_IP> \
    -e MASTER_PORT=29500 \
    ghcr.io/<your-org>/nebius-h100-16gpu-benchmark:latest \
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=<NODE_RANK> \
    --master_addr=<MASTER_PRIVATE_IP> --master_port=29500 \
    /app/scripts/benchmark.py --mode distributed
```

#### Running Directly

On Node 0 (master):
```bash
cd exercise2
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<MASTER_PRIVATE_IP> --master_port=29500 \
    scripts/benchmark.py --mode distributed --output results/benchmark_16gpu.json
```

On Node 1 (worker):
```bash
cd exercise2
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=<MASTER_PRIVATE_IP> --master_port=29500 \
    scripts/benchmark.py --mode distributed
```

### Building the Container Locally

```bash
cd exercise2

# Build the image
docker build -t gpu-cluster-test:local .

# Run tests
docker run --rm gpu-cluster-test:local pytest /app/tests -v
```

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) automates:

1. **Lint and Test**: Runs flake8 linting and pytest unit tests (CPU-only)
2. **Build**: Builds the Docker image using buildx
3. **Push**: Pushes to GitHub Container Registry (ghcr.io) on main branch
4. **Test Container**: Validates the built image runs correctly
5. **Release**: Creates GitHub releases with container tags on version tags

**Triggering the pipeline:**
- Push to `main` or `develop` branches
- Pull requests to `main`
- Git tags matching `v*` (e.g., `v1.0.0`)

**Container image naming:**
```
ghcr.io/<owner>/<repo>:latest      # Latest from main
ghcr.io/<owner>/<repo>:sha-<hash>  # Specific commit
ghcr.io/<owner>/<repo>:v1.0.0      # Release version
```

### Running on Kubernetes

Deploy the multi-node benchmark job to a Kubernetes cluster:

```bash
cd exercise2/k8s

# Multi-node benchmark (2 nodes x 8 GPUs)
kubectl apply -f benchmark-job.yaml

# Watch job progress
kubectl get jobs -w

# View logs from both pods
kubectl logs -f job/gpu-benchmark-multinode --all-containers

# Clean up
kubectl delete -f benchmark-job.yaml
```

**Resource requirements per node:**
- 8x NVIDIA GPUs (nvidia.com/gpu: 8)
- 1400 Gi memory limit
- 64 CPU cores
- InfiniBand network connectivity

---

## Demo Plan (5-7 minutes)

### 1. Repository Overview (1 min)
- Show repository structure
- Explain multi-node configuration (2 nodes x 8 GPUs)
- Highlight InfiniBand interconnect for inter-node communication

### 2. Exercise 2: Acceptance Testing Results (2 min)
```bash
# Show benchmark results
cat exercise2/results/benchmark_16gpu.json | python -m json.tool

# Highlight key metrics
# - NCCL AllReduce: 434.99 GB/s (InfiniBand)
# - Training: 137,162 tokens/s total
# - Scaling efficiency: 98%
```

### 3. Exercise 1: Training Configuration (2 min)
```bash
# Show Slurm job script
cat exercise1/scripts/train_job.sbatch

# Show training config
cat exercise1/configs/training_config.yaml

# Explain DeepSpeed ZeRO-3 config
cat exercise1/configs/ds_config_zero3.json
```

### 4. Failure Modes and Recovery (1-2 min)
- **NCCL timeout**: Check InfiniBand connectivity (`ibstat`), verify NCCL environment variables
- **OOM errors**: Reduce batch size, enable gradient checkpointing
- **Node failure**: Slurm automatically reschedules; resume from checkpoint
- **Storage issues**: Verify shared filesystem mount on both nodes

### 5. Q&A

---

## Troubleshooting

### NCCL Communication Failures (Multi-Node)
```bash
# Check InfiniBand status on both nodes
ibstat

# Verify inter-node connectivity
ping <OTHER_NODE_IP>

# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Check NCCL InfiniBand settings
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5

# Verify GPU topology
nvidia-smi topo -m
```

### Out of Memory (OOM)
- Reduce `batch_size` in training config
- Enable `gradient_checkpointing: true`
- Increase DeepSpeed ZeRO stage (currently ZeRO-3)

### Job Failures
```bash
# Check Slurm job status
scontrol show job <JOB_ID>

# View error logs
cat /mnt/shared/logs/train_<JOB_ID>.err

# Check node health
sinfo -N -l
```

---

## Benchmark Date

2025-12-15

## Author

Supreeth Mysore
