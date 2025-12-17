#!/bin/bash
# hardware_info.sh - Discover multi-node cluster hardware configuration
# Usage: ./hardware_info.sh [all|local|remote|ib|gpu]

MODE="${1:-all}"
WORKER_NODE="10.2.0.0"

print_header() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

# Local node info
local_info() {
    print_header "LOCAL NODE INFORMATION"
    echo "Hostname: $(hostname)"
    echo "IP Address: $(hostname -I | awk '{print $1}')"

    print_header "CPU"
    lscpu | grep -E "Model name|Socket|Core|Thread|CPU\(s\):" | head -6

    print_header "MEMORY"
    free -h | grep -E "Mem|Swap"

    print_header "GPU"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

    print_header "STORAGE"
    df -h | grep -E "Filesystem|/dev/" | head -5
}

# GPU topology
gpu_info() {
    print_header "GPU TOPOLOGY (NVLink Connections)"
    nvidia-smi topo -m

    print_header "NVLINK BANDWIDTH (per link)"
    nvidia-smi nvlink -s 2>/dev/null | head -20

    print_header "GPU DETAILS"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,power.draw,utilization.gpu --format=csv
}

# InfiniBand info
ib_info() {
    print_header "INFINIBAND ADAPTERS"
    ibstat 2>/dev/null | grep -E "CA |Port |State|Physical|Rate" || echo "ibstat not available"

    print_header "INFINIBAND HARDWARE (PCIe)"
    lspci | grep -i -E "mellanox|infiniband"

    print_header "INFINIBAND PERFORMANCE TEST"
    echo "To test IB bandwidth between nodes, run:"
    echo "  Node 1: ib_write_bw -d mlx5_0"
    echo "  Node 2: ib_write_bw -d mlx5_0 <node1_ip>"
}

# Remote node info
remote_info() {
    print_header "WORKER NODE ($WORKER_NODE)"

    if ssh -o ConnectTimeout=5 $WORKER_NODE "echo 'Connected'" 2>/dev/null; then
        ssh $WORKER_NODE "
            echo 'Hostname: \$(hostname)'
            echo 'IP: \$(hostname -I | awk \"{print \\\$1}\")'
            echo ''
            echo '=== CPU ==='
            lscpu | grep 'Model name'
            echo ''
            echo '=== Memory ==='
            free -h | grep Mem
            echo ''
            echo '=== GPUs ==='
            nvidia-smi --query-gpu=index,name,memory.total --format=csv
            echo ''
            echo '=== InfiniBand ==='
            ibstat 2>/dev/null | grep -E 'CA |State:|Rate:' | head -16
        "
    else
        echo "Cannot connect to worker node $WORKER_NODE"
    fi
}

# Cluster summary
cluster_summary() {
    print_header "CLUSTER SUMMARY"

    echo "Node 0 (Master): $(hostname) - $(hostname -I | awk '{print $1}')"

    if ssh -o ConnectTimeout=5 $WORKER_NODE "hostname" 2>/dev/null; then
        WORKER_HOSTNAME=$(ssh $WORKER_NODE "hostname")
        echo "Node 1 (Worker): $WORKER_HOSTNAME - $WORKER_NODE"
    else
        echo "Node 1 (Worker): Unknown - $WORKER_NODE (not reachable)"
    fi

    echo ""
    echo "Per Node:"
    echo "  - CPU: Intel Xeon Platinum 8468 (128 cores)"
    echo "  - RAM: 1.5 TB"
    echo "  - GPUs: 8x NVIDIA H100 80GB HBM3"
    echo "  - InfiniBand: 8x Mellanox ConnectX (400 Gb/s each)"
    echo ""
    echo "Total Cluster:"
    echo "  - Nodes: 2"
    echo "  - Total GPUs: 16"
    echo "  - Total GPU Memory: 1.28 TB"
    echo "  - Total RAM: 3 TB"
    echo "  - Interconnect: NVLink (intra-node), InfiniBand NDR (inter-node)"
}

# Main
case $MODE in
    all)
        cluster_summary
        local_info
        gpu_info
        ib_info
        remote_info
        ;;
    local)
        local_info
        ;;
    remote)
        remote_info
        ;;
    ib)
        ib_info
        ;;
    gpu)
        gpu_info
        ;;
    summary)
        cluster_summary
        ;;
    *)
        echo "Usage: ./hardware_info.sh [all|local|remote|ib|gpu|summary]"
        echo ""
        echo "Options:"
        echo "  all     - Full hardware info (default)"
        echo "  local   - Local node only"
        echo "  remote  - Worker node only"
        echo "  ib      - InfiniBand info"
        echo "  gpu     - GPU topology and status"
        echo "  summary - Quick cluster summary"
        ;;
esac
