#!/bin/bash
# tmux_demo.sh - Create a multi-pane tmux layout for demo
# Usage: ./tmux_demo.sh [start|attach|kill]

SESSION_NAME="gpu_demo"
REPO_PATH="/home/supreethlab/repos/nebius-h100-16gpu-benchmark"

start_session() {
    # Kill existing session if exists
    tmux kill-session -t $SESSION_NAME 2>/dev/null

    # Create new session
    tmux new-session -d -s $SESSION_NAME -x 200 -y 50

    # Rename first window
    tmux rename-window -t $SESSION_NAME:0 'Demo'

    # Split into 4 panes
    # Layout:
    # ┌─────────────────┬─────────────────┐
    # │  GPU Status     │  Training Logs  │
    # │  (gpustat)      │                 │
    # ├─────────────────┼─────────────────┤
    # │  GPU Metrics    │  Commands       │
    # │  (nvidia-smi)   │                 │
    # └─────────────────┴─────────────────┘

    # Split horizontally (left | right)
    tmux split-window -h -t $SESSION_NAME:0

    # Split left pane vertically (top-left | bottom-left)
    tmux split-window -v -t $SESSION_NAME:0.0

    # Split right pane vertically (top-right | bottom-right)
    tmux split-window -v -t $SESSION_NAME:0.2

    # Set up each pane
    # Pane 0 (top-left): GPU status with gpustat
    tmux send-keys -t $SESSION_NAME:0.0 "cd $REPO_PATH && source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-finetune && gpustat -i 1 --color" C-m

    # Pane 1 (bottom-left): nvidia-smi dmon
    tmux send-keys -t $SESSION_NAME:0.1 "watch -n 2 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv'" C-m

    # Pane 2 (top-right): Ready for training logs
    tmux send-keys -t $SESSION_NAME:0.2 "cd $REPO_PATH && echo 'Ready for training logs. Run: source multinode.sh exercise1'" C-m

    # Pane 3 (bottom-right): Command pane
    tmux send-keys -t $SESSION_NAME:0.3 "cd $REPO_PATH && source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-finetune && ./demo_banner.sh" C-m

    # Select the command pane (bottom-right)
    tmux select-pane -t $SESSION_NAME:0.3

    echo -e "\033[1;32mDemo session started!\033[0m"
    echo ""
    echo "Layout:"
    echo "┌─────────────────┬─────────────────┐"
    echo "│  GPU Status     │  Training Logs  │"
    echo "│  (gpustat)      │                 │"
    echo "├─────────────────┼─────────────────┤"
    echo "│  GPU Metrics    │  Commands       │"
    echo "│  (nvidia-smi)   │                 │"
    echo "└─────────────────┴─────────────────┘"
    echo ""
    echo "Commands:"
    echo "  Attach:  tmux attach -t $SESSION_NAME"
    echo "  Detach:  Ctrl+B, then D"
    echo "  Kill:    ./tmux_demo.sh kill"
    echo ""
    echo "Pane navigation: Ctrl+B, then arrow keys"
}

attach_session() {
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        tmux attach -t $SESSION_NAME
    else
        echo "Session '$SESSION_NAME' does not exist. Run './tmux_demo.sh start' first."
    fi
}

kill_session() {
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        tmux kill-session -t $SESSION_NAME
        echo "Session '$SESSION_NAME' killed."
    else
        echo "Session '$SESSION_NAME' does not exist."
    fi
}

case "${1:-start}" in
    start)
        start_session
        echo -e "\033[1;33mRun 'tmux attach -t $SESSION_NAME' to view the demo\033[0m"
        ;;
    attach)
        attach_session
        ;;
    kill)
        kill_session
        ;;
    *)
        echo "Usage: ./tmux_demo.sh [start|attach|kill]"
        echo ""
        echo "Commands:"
        echo "  start   - Create demo tmux session with 4 panes"
        echo "  attach  - Attach to existing demo session"
        echo "  kill    - Kill the demo session"
        ;;
esac
