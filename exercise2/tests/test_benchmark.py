#!/usr/bin/env python3
"""
test_benchmark.py - Unit tests for GPU cluster benchmark

These tests can run without GPU (CPU-only mode) for CI validation.
"""

import json
import pytest
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestBenchmarkImports:
    """Test that all imports work correctly."""
    
    def test_import_benchmark(self):
        """Test benchmark module can be imported."""
        import benchmark
        assert hasattr(benchmark, 'run_benchmarks')
        assert hasattr(benchmark, 'check_gpu_health')
        assert hasattr(benchmark, 'SimpleTransformer')
    
    def test_import_torch(self):
        """Test PyTorch is available."""
        import torch
        assert torch.__version__


class TestDataClasses:
    """Test data class definitions."""
    
    def test_gpu_info(self):
        """Test GPUInfo dataclass."""
        from benchmark import GPUInfo
        
        info = GPUInfo(
            index=0,
            name="Test GPU",
            memory_total_gb=80.0,
            memory_free_gb=79.0,
            cuda_version="12.4",
            driver_version="550.54",
        )
        
        assert info.index == 0
        assert info.name == "Test GPU"
        assert info.memory_total_gb == 80.0
    
    def test_benchmark_result(self):
        """Test BenchmarkResult dataclass."""
        from benchmark import BenchmarkResult
        
        result = BenchmarkResult(
            name="test_benchmark",
            status="PASS",
            metric_name="throughput",
            metric_value=100.0,
            metric_unit="ops/s",
            duration_seconds=1.5,
            details={"test": True},
        )
        
        assert result.name == "test_benchmark"
        assert result.status == "PASS"
        assert result.metric_value == 100.0
    
    def test_cluster_info(self):
        """Test ClusterInfo dataclass."""
        from benchmark import ClusterInfo
        
        info = ClusterInfo(
            hostname="test-node",
            nodes=2,
            gpus_per_node=8,
            total_gpus=16,
            gpu_model="H100",
            cuda_version="12.4",
            pytorch_version="2.3.0",
            nccl_version="2.21.5",
        )
        
        assert info.total_gpus == 16
        assert info.nodes == 2


class TestSimpleTransformer:
    """Test the transformer model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        import torch
        from benchmark import SimpleTransformer
        
        model = SimpleTransformer(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            max_seq_length=64,
        )
        
        assert model is not None
    
    def test_model_forward_cpu(self):
        """Test model forward pass on CPU."""
        import torch
        from benchmark import SimpleTransformer
        
        model = SimpleTransformer(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            max_seq_length=64,
        )
        
        # Create input
        x = torch.randint(0, 1000, (2, 32))
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (2, 32, 1000)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_get_cluster_info(self):
        """Test cluster info collection."""
        from benchmark import get_cluster_info
        
        info = get_cluster_info()
        
        assert info.hostname
        assert info.pytorch_version
    
    def test_print_results(self):
        """Test results printing doesn't crash."""
        from benchmark import print_results
        
        results = {
            "timestamp": "2024-01-01T00:00:00Z",
            "cluster_info": {
                "hostname": "test",
                "nodes": 1,
                "gpus_per_node": 0,
                "total_gpus": 0,
                "gpu_model": "N/A",
                "cuda_version": "N/A",
                "pytorch_version": "2.3.0",
                "nccl_version": "N/A",
            },
            "results": {
                "test": {
                    "status": "PASS",
                    "metric_value": 100,
                    "metric_unit": "ops/s",
                }
            },
            "overall_status": "PASS",
        }
        
        # Should not raise
        print_results(results)


class TestGPUHealthCheck:
    """Test GPU health check function."""
    
    def test_health_check_no_gpu(self):
        """Test health check handles no GPU gracefully."""
        import torch
        from benchmark import check_gpu_health
        
        healthy, gpu_infos, message = check_gpu_health()
        
        # Result depends on whether CUDA is available
        if torch.cuda.is_available():
            assert healthy
            assert len(gpu_infos) > 0
        else:
            assert not healthy
            assert len(gpu_infos) == 0


class TestResultsSerialization:
    """Test that results can be serialized to JSON."""
    
    def test_serialize_benchmark_result(self):
        """Test BenchmarkResult can be serialized."""
        from dataclasses import asdict
        from benchmark import BenchmarkResult
        
        result = BenchmarkResult(
            name="test",
            status="PASS",
            metric_name="value",
            metric_value=42.0,
            metric_unit="units",
            duration_seconds=1.0,
            details={"key": "value"},
        )
        
        # Convert to dict and serialize
        result_dict = asdict(result)
        json_str = json.dumps(result_dict)
        
        # Deserialize and verify
        loaded = json.loads(json_str)
        assert loaded["name"] == "test"
        assert loaded["status"] == "PASS"
        assert loaded["metric_value"] == 42.0


# GPU-specific tests (skipped if no GPU)
@pytest.mark.skipif(
    not os.environ.get("TEST_WITH_GPU"),
    reason="GPU tests require TEST_WITH_GPU=1"
)
class TestGPUBenchmarks:
    """Tests that require GPU."""
    
    def test_matmul_benchmark(self):
        """Test matrix multiplication benchmark."""
        import torch
        from benchmark import benchmark_matmul
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device("cuda:0")
        result = benchmark_matmul(
            device,
            sizes=[1024],  # Smaller size for testing
            iterations=10,
            warmup=2,
        )
        
        assert result.name == "matmul_benchmark"
        assert result.metric_value > 0
    
    def test_memory_bandwidth_benchmark(self):
        """Test memory bandwidth benchmark."""
        import torch
        from benchmark import benchmark_memory_bandwidth
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device("cuda:0")
        result = benchmark_memory_bandwidth(
            device,
            size_gb=0.1,  # Smaller size for testing
            iterations=10,
            warmup=2,
        )
        
        assert result.name == "memory_bandwidth"
        assert result.metric_value > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
