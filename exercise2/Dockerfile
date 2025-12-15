# Dockerfile - GPU Cluster Acceptance Testing Container
#
# Based on NVIDIA PyTorch container with pre-installed CUDA, cuDNN, NCCL
# Includes distributed training benchmarks for GPU cluster validation

FROM nvcr.io/nvidia/pytorch:24.07-py3

LABEL maintainer="ML Team"
LABEL description="GPU Cluster Acceptance Testing Solution"
LABEL version="1.0.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    htop \
    iotop \
    net-tools \
    iputils-ping \
    dnsutils \
    infiniband-diags \
    ibverbs-utils \
    perftest \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Dependencies
# =============================================================================

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install ML/benchmark dependencies
RUN pip install --no-cache-dir \
    transformers==4.44.0 \
    datasets==2.20.0 \
    accelerate==0.33.0 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    pytest==8.2.2 \
    pytest-timeout==2.3.1 \
    psutil==5.9.8 \
    py3nvml==0.2.7 \
    tabulate==0.9.0

# =============================================================================
# Copy Application Files
# =============================================================================

# Copy scripts and configs
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/
COPY tests/ /app/tests/

# Make scripts executable
RUN chmod +x /app/scripts/*.py

# =============================================================================
# Environment Configuration
# =============================================================================

# NCCL settings for optimal performance
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=5
ENV NCCL_P2P_LEVEL=NVL

# PyTorch distributed settings
ENV TORCH_DISTRIBUTED_DEBUG=OFF
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# CUDA settings
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Application settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# Default Entrypoint
# =============================================================================

# Default command runs the health check
ENTRYPOINT ["python", "/app/scripts/benchmark.py"]
CMD ["--mode", "health"]

# =============================================================================
# Health Check
# =============================================================================

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1
