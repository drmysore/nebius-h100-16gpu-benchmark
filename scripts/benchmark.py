#!/usr/bin/env python3
"""
benchmark.py - GPU Cluster Acceptance Testing Benchmarks

This script provides comprehensive GPU cluster testing including:
1. GPU health checks
2. Single-GPU performance benchmarks
3. Multi-GPU distributed benchmarks (NCCL, DDP)
4. End-to-end training benchmark

Usage:
    # Health check only
    python benchmark.py --mode health
    
    # Single GPU benchmark
    python benchmark.py --mode single
    
    # Distributed benchmark (use with torchrun)
    torchrun --nproc_per_node=8 benchmark.py --mode distributed
    
    # Full test suite
    python benchmark.py --mode full
"""

import argparse
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    cuda_version: str
    driver_version: str


@dataclass
class BenchmarkResult:
    name: str
    status: str  # PASS, FAIL, SKIP
    metric_name: str
    metric_value: float
    metric_unit: str
    duration_seconds: float
    details: Dict[str, Any]


@dataclass
class ClusterInfo:
    hostname: str
    nodes: int
    gpus_per_node: int
    total_gpus: int
    gpu_model: str
    cuda_version: str
    pytorch_version: str
    nccl_version: str


# =============================================================================
# GPU Health Check
# =============================================================================

def check_gpu_health() -> Tuple[bool, List[GPUInfo], str]:
    """Check GPU availability and basic health."""
    if not torch.cuda.is_available():
        return False, [], "CUDA not available"
    
    gpu_infos = []
    try:
        num_gpus = torch.cuda.device_count()
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / (1024**3)
            memory_free = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3)
            
            gpu_info = GPUInfo(
                index=i,
                name=props.name,
                memory_total_gb=round(memory_total, 2),
                memory_free_gb=round(memory_free, 2),
                cuda_version=torch.version.cuda,
                driver_version=torch.cuda.get_device_capability(i).__str__(),
            )
            gpu_infos.append(gpu_info)
            
            # Basic tensor operation test
            with torch.cuda.device(i):
                x = torch.randn(1000, 1000, device=f"cuda:{i}")
                y = torch.matmul(x, x)
                del x, y
                torch.cuda.empty_cache()
        
        return True, gpu_infos, "All GPUs healthy"
        
    except Exception as e:
        return False, gpu_infos, f"GPU health check failed: {str(e)}"


# =============================================================================
# Matrix Multiplication Benchmark
# =============================================================================

def benchmark_matmul(
    device: torch.device,
    sizes: List[int] = [4096, 8192, 16384],
    dtype: torch.dtype = torch.bfloat16,
    iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """Benchmark matrix multiplication throughput."""
    start_time = time.time()
    results = []
    
    try:
        for size in sizes:
            # Create matrices
            a = torch.randn(size, size, dtype=dtype, device=device)
            b = torch.randn(size, size, dtype=dtype, device=device)
            
            # Warmup
            for _ in range(warmup):
                torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(iterations):
                torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # Calculate TFLOPS (2 * N^3 for matrix multiplication)
            flops = 2 * (size ** 3) * iterations
            tflops = flops / elapsed / 1e12
            
            results.append({
                "size": size,
                "tflops": round(tflops, 2),
                "time_per_op_ms": round(elapsed / iterations * 1000, 3),
            })
            
            del a, b
            torch.cuda.empty_cache()
        
        # Use largest size for main metric
        best_tflops = max(r["tflops"] for r in results)
        status = "PASS" if best_tflops > 100 else "FAIL"  # Minimum threshold
        
        return BenchmarkResult(
            name="matmul_benchmark",
            status=status,
            metric_name="tflops",
            metric_value=best_tflops,
            metric_unit="TFLOPS",
            duration_seconds=round(time.time() - start_time, 2),
            details={"results_by_size": results, "dtype": str(dtype)},
        )
        
    except Exception as e:
        return BenchmarkResult(
            name="matmul_benchmark",
            status="FAIL",
            metric_name="tflops",
            metric_value=0,
            metric_unit="TFLOPS",
            duration_seconds=round(time.time() - start_time, 2),
            details={"error": str(e)},
        )


# =============================================================================
# Memory Bandwidth Benchmark
# =============================================================================

def benchmark_memory_bandwidth(
    device: torch.device,
    size_gb: float = 1.0,
    iterations: int = 50,
    warmup: int = 5,
) -> BenchmarkResult:
    """Benchmark GPU memory bandwidth."""
    start_time = time.time()
    
    try:
        # Calculate tensor size
        num_elements = int(size_gb * 1024**3 / 4)  # float32 = 4 bytes
        
        # Create tensors
        src = torch.randn(num_elements, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)
        
        # Warmup
        for _ in range(warmup):
            dst.copy_(src)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate bandwidth (read + write)
        bytes_transferred = num_elements * 4 * 2 * iterations  # float32, read+write
        bandwidth_tbps = bytes_transferred / elapsed / 1e12
        
        del src, dst
        torch.cuda.empty_cache()
        
        status = "PASS" if bandwidth_tbps > 1.0 else "FAIL"  # Minimum 1 TB/s
        
        return BenchmarkResult(
            name="memory_bandwidth",
            status=status,
            metric_name="bandwidth",
            metric_value=round(bandwidth_tbps, 2),
            metric_unit="TB/s",
            duration_seconds=round(time.time() - start_time, 2),
            details={"size_gb": size_gb, "iterations": iterations},
        )
        
    except Exception as e:
        return BenchmarkResult(
            name="memory_bandwidth",
            status="FAIL",
            metric_name="bandwidth",
            metric_value=0,
            metric_unit="TB/s",
            duration_seconds=round(time.time() - start_time, 2),
            details={"error": str(e)},
        )


# =============================================================================
# NCCL Benchmark
# =============================================================================

def benchmark_nccl(
    sizes_mb: List[int] = [1, 10, 100, 1000],
    iterations: int = 50,
    warmup: int = 5,
) -> BenchmarkResult:
    """Benchmark NCCL collective operations."""
    start_time = time.time()
    
    if not dist.is_initialized():
        return BenchmarkResult(
            name="nccl_benchmark",
            status="SKIP",
            metric_name="bandwidth",
            metric_value=0,
            metric_unit="GB/s",
            duration_seconds=0,
            details={"reason": "Distributed not initialized"},
        )
    
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        
        results = []
        
        for size_mb in sizes_mb:
            num_elements = size_mb * 1024 * 1024 // 4  # float32
            tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
            
            # Warmup
            for _ in range(warmup):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(iterations):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # Calculate bandwidth
            # AllReduce: 2 * (n-1) / n * data_size for ring algorithm
            algo_bandwidth = (2 * (world_size - 1) / world_size) * (size_mb * 1024**2)
            bandwidth_gbps = algo_bandwidth * iterations / elapsed / 1e9
            
            results.append({
                "size_mb": size_mb,
                "bandwidth_gbps": round(bandwidth_gbps, 2),
                "latency_ms": round(elapsed / iterations * 1000, 3),
            })
            
            del tensor
            torch.cuda.empty_cache()
        
        # Use largest size for main metric
        best_bandwidth = max(r["bandwidth_gbps"] for r in results)
        status = "PASS" if best_bandwidth > 10 else "FAIL"  # Minimum 10 GB/s
        
        return BenchmarkResult(
            name="nccl_benchmark",
            status=status,
            metric_name="all_reduce_bandwidth",
            metric_value=best_bandwidth,
            metric_unit="GB/s",
            duration_seconds=round(time.time() - start_time, 2),
            details={
                "world_size": world_size,
                "results_by_size": results,
            },
        )
        
    except Exception as e:
        return BenchmarkResult(
            name="nccl_benchmark",
            status="FAIL",
            metric_name="all_reduce_bandwidth",
            metric_value=0,
            metric_unit="GB/s",
            duration_seconds=round(time.time() - start_time, 2),
            details={"error": str(e)},
        )


# =============================================================================
# Training Benchmark (Simple Transformer)
# =============================================================================

class SimpleTransformer(nn.Module):
    """Simple transformer model for benchmarking."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) + self.position_embedding(positions)
        x = self.transformer(x)
        x = self.output(x)
        return x


def benchmark_training(
    batch_size: int = 8,
    seq_length: int = 512,
    num_steps: int = 50,
    warmup_steps: int = 5,
) -> BenchmarkResult:
    """Benchmark distributed training throughput."""
    start_time = time.time()
    
    try:
        # Setup device
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = rank % torch.cuda.device_count()
        else:
            rank = 0
            world_size = 1
            local_rank = 0
        
        device = torch.device(f"cuda:{local_rank}")
        
        # Create model
        model = SimpleTransformer().to(device)
        
        if dist.is_initialized():
            model = DDP(model, device_ids=[local_rank])
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Generate random data
        def get_batch():
            x = torch.randint(0, 50257, (batch_size, seq_length), device=device)
            y = torch.randint(0, 50257, (batch_size, seq_length), device=device)
            return x, y
        
        # Warmup
        for _ in range(warmup_steps):
            x, y = get_batch()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, 50257), y.view(-1))
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        
        # Benchmark
        total_tokens = 0
        start = time.time()
        
        for step in range(num_steps):
            x, y = get_batch()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, 50257), y.view(-1))
            loss.backward()
            optimizer.step()
            total_tokens += batch_size * seq_length * world_size
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate throughput
        tokens_per_second = total_tokens / elapsed
        tokens_per_gpu = tokens_per_second / world_size
        
        status = "PASS" if tokens_per_gpu > 500 else "FAIL"  # Minimum 500 tokens/s/GPU
        
        return BenchmarkResult(
            name="training_benchmark",
            status=status,
            metric_name="tokens_per_second",
            metric_value=round(tokens_per_second, 1),
            metric_unit="tokens/s",
            duration_seconds=round(time.time() - start_time, 2),
            details={
                "tokens_per_gpu": round(tokens_per_gpu, 1),
                "batch_size": batch_size,
                "seq_length": seq_length,
                "world_size": world_size,
                "num_steps": num_steps,
            },
        )
        
    except Exception as e:
        return BenchmarkResult(
            name="training_benchmark",
            status="FAIL",
            metric_name="tokens_per_second",
            metric_value=0,
            metric_unit="tokens/s",
            duration_seconds=round(time.time() - start_time, 2),
            details={"error": str(e)},
        )


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def get_cluster_info() -> ClusterInfo:
    """Collect cluster information."""
    hostname = socket.gethostname()
    
    if dist.is_initialized():
        world_size = dist.get_world_size()
        # Estimate nodes (assume 8 GPUs per node)
        gpus_per_node = min(8, torch.cuda.device_count())
        nodes = max(1, world_size // gpus_per_node)
    else:
        world_size = torch.cuda.device_count()
        gpus_per_node = world_size
        nodes = 1
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    
    # Get NCCL version
    try:
        nccl_version = ".".join(map(str, torch.cuda.nccl.version()))
    except:
        nccl_version = "N/A"
    
    return ClusterInfo(
        hostname=hostname,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        total_gpus=world_size,
        gpu_model=gpu_name,
        cuda_version=torch.version.cuda or "N/A",
        pytorch_version=torch.__version__,
        nccl_version=nccl_version,
    )


def run_benchmarks(mode: str) -> Dict[str, Any]:
    """Run benchmark suite based on mode."""
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cluster_info": None,
        "results": {},
        "overall_status": "PASS",
    }
    
    # Get cluster info
    results["cluster_info"] = asdict(get_cluster_info())
    
    # Determine rank for logging
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # GPU Health Check
    if mode in ["health", "single", "full"]:
        if rank == 0:
            logger.info("Running GPU health check...")
        healthy, gpu_infos, message = check_gpu_health()
        results["results"]["gpu_health"] = {
            "status": "PASS" if healthy else "FAIL",
            "message": message,
            "gpus": [asdict(g) for g in gpu_infos],
        }
        if not healthy:
            results["overall_status"] = "FAIL"
    
    # Single GPU benchmarks
    if mode in ["single", "full"]:
        device = torch.device("cuda:0")
        
        if rank == 0:
            logger.info("Running matrix multiplication benchmark...")
        matmul_result = benchmark_matmul(device)
        results["results"]["matmul_benchmark"] = asdict(matmul_result)
        if matmul_result.status == "FAIL":
            results["overall_status"] = "FAIL"
        
        if rank == 0:
            logger.info("Running memory bandwidth benchmark...")
        membw_result = benchmark_memory_bandwidth(device)
        results["results"]["memory_bandwidth"] = asdict(membw_result)
        if membw_result.status == "FAIL":
            results["overall_status"] = "FAIL"
    
    # Distributed benchmarks
    if mode in ["distributed", "full"]:
        if rank == 0:
            logger.info("Running NCCL benchmark...")
        nccl_result = benchmark_nccl()
        results["results"]["nccl_benchmark"] = asdict(nccl_result)
        if nccl_result.status == "FAIL":
            results["overall_status"] = "FAIL"
        
        if rank == 0:
            logger.info("Running training benchmark...")
        training_result = benchmark_training()
        results["results"]["training_benchmark"] = asdict(training_result)
        if training_result.status == "FAIL":
            results["overall_status"] = "FAIL"
    
    return results


def print_results(results: Dict[str, Any]):
    """Print results in a formatted way."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank != 0:
        return
    
    print("\n" + "=" * 70)
    print("GPU CLUSTER BENCHMARK RESULTS")
    print("=" * 70)
    
    # Cluster info
    info = results["cluster_info"]
    print(f"\nCluster: {info['hostname']}")
    print(f"  Nodes: {info['nodes']}")
    print(f"  GPUs per node: {info['gpus_per_node']}")
    print(f"  Total GPUs: {info['total_gpus']}")
    print(f"  GPU Model: {info['gpu_model']}")
    print(f"  CUDA: {info['cuda_version']}")
    print(f"  PyTorch: {info['pytorch_version']}")
    print(f"  NCCL: {info['nccl_version']}")
    
    # Results
    print("\nBenchmark Results:")
    print("-" * 70)
    
    for name, result in results["results"].items():
        if isinstance(result, dict) and "status" in result:
            status = result["status"]
            status_symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "○"
            
            if "metric_value" in result:
                print(f"  {status_symbol} {name}: {result['metric_value']} {result.get('metric_unit', '')}")
            else:
                print(f"  {status_symbol} {name}: {result.get('message', status)}")
    
    print("-" * 70)
    overall = results["overall_status"]
    print(f"\nOverall Status: {'✓ PASS' if overall == 'PASS' else '✗ FAIL'}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="GPU Cluster Benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["health", "single", "distributed", "full"],
        default="health",
        help="Benchmark mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()
    
    # Initialize distributed if environment variables are set
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        try:
            dist.init_process_group(backend="nccl")
            logger.info(f"Initialized distributed: rank {dist.get_rank()}/{dist.get_world_size()}")
        except Exception as e:
            logger.warning(f"Failed to initialize distributed: {e}")
    
    # Run benchmarks
    results = run_benchmarks(args.mode)
    
    # Print results
    print_results(results)
    
    # Save to file
    rank = dist.get_rank() if dist.is_initialized() else 0
    if args.output and rank == 0:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
