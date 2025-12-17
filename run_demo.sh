#!/bin/bash
# run_demo.sh - Full interactive demo sequence for 16-GPU cluster
# Usage: ./run_demo.sh [full|quick|benchmark|training]

REPO_PATH="/home/supreethlab/repos/nebius-h100-16gpu-benchmark"
MODE="${1:-full}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

print_step() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}[$1/$2]${NC} ${GREEN}$3${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

wait_for_user() {
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read -r
}

cd $REPO_PATH

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-finetune

run_full_demo() {
    TOTAL_STEPS=6

    # Step 1: Banner
    print_step 1 $TOTAL_STEPS "Welcome to Nebius H100 GPU Cluster Demo"
    ./demo_banner.sh
    wait_for_user

    # Step 2: Cluster Hardware
    print_step 2 $TOTAL_STEPS "Cluster Hardware Configuration"
    ./hardware_info.sh summary
    wait_for_user

    # Step 3: GPU Topology
    print_step 3 $TOTAL_STEPS "GPU Topology and NVLink Connections"
    ./hardware_info.sh gpu
    wait_for_user

    # Step 4: Health Check
    print_step 4 $TOTAL_STEPS "GPU Health Check"
    source multinode.sh exercise2 health
    wait_for_user

    # Step 5: Benchmarks
    print_step 5 $TOTAL_STEPS "Running GPU Benchmarks (16 GPUs)"
    echo "This will run distributed benchmarks across both nodes..."
    wait_for_user
    source multinode.sh exercise2 distributed
    wait_for_user

    # Step 6: Training
    print_step 6 $TOTAL_STEPS "LLM Fine-Tuning Demo (16 GPUs)"
    echo "Starting Qwen2-7B fine-tuning with LoRA..."
    echo "Configuration:"
    echo "  - Model: Qwen/Qwen2-7B-Instruct"
    echo "  - Method: LoRA (rank=64)"
    echo "  - GPUs: 16 (2 nodes x 8 GPUs)"
    echo "  - Steps: 100"
    wait_for_user
    source multinode.sh exercise1
    wait_for_user

    # Summary
    print_step "DONE" "" "Demo Complete"
    echo -e "${GREEN}Results Summary:${NC}"
    echo ""
    echo "Benchmarks:"
    ls -la exercise2/results/*.json 2>/dev/null | tail -3
    echo ""
    echo "Training Checkpoints:"
    ls -la /home/supreethlab/training/checkpoints/ 2>/dev/null | head -5
    echo ""
    echo -e "${CYAN}Thank you for attending the demo!${NC}"
}

run_quick_demo() {
    TOTAL_STEPS=3

    print_step 1 $TOTAL_STEPS "Cluster Overview"
    ./demo_banner.sh
    ./hardware_info.sh summary
    wait_for_user

    print_step 2 $TOTAL_STEPS "Quick Benchmark"
    source multinode.sh exercise2 health
    wait_for_user

    print_step 3 $TOTAL_STEPS "Demo Complete"
    echo "For full training demo, run: ./run_demo.sh full"
}

run_benchmark_only() {
    print_step 1 2 "Running Benchmarks"
    ./demo_banner.sh
    source multinode.sh exercise2

    print_step 2 2 "Benchmark Results"
    echo "Results saved to exercise2/results/"
    ls -la exercise2/results/*.json 2>/dev/null | tail -5
}

run_training_only() {
    print_step 1 2 "Starting Training"
    ./demo_banner.sh
    echo "Configuration: 16 GPUs, Qwen2-7B, LoRA"
    wait_for_user
    source multinode.sh exercise1

    print_step 2 2 "Training Complete"
    echo "Checkpoints saved to /home/supreethlab/training/checkpoints/"
    ls -la /home/supreethlab/training/checkpoints/ 2>/dev/null | head -5
}

# Main
case $MODE in
    full)
        run_full_demo
        ;;
    quick)
        run_quick_demo
        ;;
    benchmark)
        run_benchmark_only
        ;;
    training)
        run_training_only
        ;;
    *)
        echo "Usage: ./run_demo.sh [full|quick|benchmark|training]"
        echo ""
        echo "Modes:"
        echo "  full      - Complete demo with all steps (15-20 min)"
        echo "  quick     - Quick overview and health check (5 min)"
        echo "  benchmark - Run benchmarks only"
        echo "  training  - Run training only"
        ;;
esac
