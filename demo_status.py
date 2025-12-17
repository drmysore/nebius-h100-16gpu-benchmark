#!/usr/bin/env python3
"""
demo_status.py - Rich terminal GPU dashboard for demo
Usage: python demo_status.py
"""

import subprocess
import time
import sys
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
except ImportError:
    print("Please install rich: pip install rich")
    sys.exit(1)

console = Console()

def get_gpu_info():
    """Get GPU information from nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip().split('\n')
    except Exception as e:
        return []

def get_cpu_memory():
    """Get CPU and memory info"""
    try:
        mem_result = subprocess.run(['free', '-h'], capture_output=True, text=True, timeout=5)
        lines = mem_result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            return f"{parts[2]} / {parts[1]}"
    except:
        pass
    return "N/A"

def get_hostname():
    """Get hostname"""
    try:
        result = subprocess.run(['hostname'], capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except:
        return "Unknown"

def create_gpu_table():
    """Create GPU status table"""
    table = Table(
        title="[bold cyan]NVIDIA H100 GPU Status[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("GPU", style="cyan", justify="center", width=6)
    table.add_column("Util", style="green", justify="center", width=8)
    table.add_column("Memory", style="yellow", justify="center", width=18)
    table.add_column("Temp", style="red", justify="center", width=8)
    table.add_column("Power", style="magenta", justify="center", width=10)
    table.add_column("Status", style="white", justify="center", width=10)

    gpu_lines = get_gpu_info()

    for line in gpu_lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 6:
            idx = parts[0]
            util = int(float(parts[1])) if parts[1] else 0
            mem_used = int(float(parts[2])) if parts[2] else 0
            mem_total = int(float(parts[3])) if parts[3] else 1
            temp = int(float(parts[4])) if parts[4] else 0
            power = float(parts[5]) if parts[5] else 0

            # Determine status based on utilization
            if util > 80:
                status = "[bold green]ACTIVE[/bold green]"
            elif util > 0:
                status = "[yellow]WORKING[/yellow]"
            else:
                status = "[dim]IDLE[/dim]"

            # Color coding for utilization
            if util > 80:
                util_str = f"[bold green]{util}%[/bold green]"
            elif util > 50:
                util_str = f"[yellow]{util}%[/yellow]"
            else:
                util_str = f"[dim]{util}%[/dim]"

            # Memory bar
            mem_pct = (mem_used / mem_total) * 100 if mem_total > 0 else 0
            mem_str = f"{mem_used//1024}G / {mem_total//1024}G"

            # Temperature color
            if temp > 70:
                temp_str = f"[bold red]{temp}C[/bold red]"
            elif temp > 50:
                temp_str = f"[yellow]{temp}C[/yellow]"
            else:
                temp_str = f"[green]{temp}C[/green]"

            table.add_row(
                f"[bold]{idx}[/bold]",
                util_str,
                mem_str,
                temp_str,
                f"{power:.0f}W",
                status
            )

    return table

def create_summary_panel():
    """Create summary panel"""
    gpu_lines = get_gpu_info()
    num_gpus = len([l for l in gpu_lines if l.strip()])

    total_util = 0
    total_mem_used = 0
    total_mem = 0
    total_power = 0

    for line in gpu_lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 6:
            total_util += int(float(parts[1])) if parts[1] else 0
            total_mem_used += int(float(parts[2])) if parts[2] else 0
            total_mem += int(float(parts[3])) if parts[3] else 0
            total_power += float(parts[5]) if parts[5] else 0

    avg_util = total_util / num_gpus if num_gpus > 0 else 0

    summary = Text()
    summary.append("Cluster Summary\n", style="bold cyan")
    summary.append(f"Node: ", style="dim")
    summary.append(f"{get_hostname()}\n", style="white")
    summary.append(f"GPUs: ", style="dim")
    summary.append(f"{num_gpus} x H100 80GB\n", style="green")
    summary.append(f"Avg Util: ", style="dim")
    summary.append(f"{avg_util:.1f}%\n", style="yellow")
    summary.append(f"Memory: ", style="dim")
    summary.append(f"{total_mem_used//1024}G / {total_mem//1024}G\n", style="yellow")
    summary.append(f"Power: ", style="dim")
    summary.append(f"{total_power:.0f}W\n", style="magenta")
    summary.append(f"Host RAM: ", style="dim")
    summary.append(f"{get_cpu_memory()}\n", style="cyan")
    summary.append(f"\nUpdated: ", style="dim")
    summary.append(f"{datetime.now().strftime('%H:%M:%S')}", style="white")

    return Panel(summary, title="[bold]Cluster Status[/bold]", border_style="cyan")

def create_dashboard():
    """Create full dashboard layout"""
    layout = Layout()
    layout.split_row(
        Layout(create_gpu_table(), name="gpus", ratio=3),
        Layout(create_summary_panel(), name="summary", ratio=1)
    )
    return layout

def main():
    console.clear()
    console.print("\n[bold cyan]Nebius H100 GPU Cluster Monitor[/bold cyan]")
    console.print("[dim]Press Ctrl+C to exit[/dim]\n")

    try:
        with Live(create_dashboard(), refresh_per_second=1, console=console) as live:
            while True:
                time.sleep(1)
                live.update(create_dashboard())
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")

if __name__ == "__main__":
    main()
