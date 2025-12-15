# variables.tf - Variable definitions for Nebius Soperator deployment

# =============================================================================
# Project Configuration
# =============================================================================

variable "folder_id" {
  description = "Nebius Folder ID"
  type        = string
}

variable "region" {
  description = "Nebius region"
  type        = string
  default     = "eu-north1"
}

# =============================================================================
# Cluster Configuration
# =============================================================================

variable "slurm_cluster_name" {
  description = "Name for the Slurm cluster"
  type        = string
  default     = "llm-finetune-poc"
}

variable "slurm_partition_name" {
  description = "Name for the Slurm partition"
  type        = string
  default     = "gpu"
}

# =============================================================================
# GPU Worker Configuration
# =============================================================================

variable "slurm_worker_count" {
  description = "Number of GPU worker nodes (each with 8x H100)"
  type        = number
  default     = 2  # 2 nodes × 8 GPUs = 16 H100 GPUs total
  
  validation {
    condition     = var.slurm_worker_count >= 1 && var.slurm_worker_count <= 64
    error_message = "Worker count must be between 1 and 64."
  }
}

variable "slurm_worker_preset" {
  description = "Worker node preset (GPU configuration)"
  type        = string
  default     = "gpu-h100-sxm_8gpu-128vcpu-1600gb"
  
  validation {
    condition = contains([
      "gpu-h100-sxm_8gpu-128vcpu-1600gb",
      "gpu-h100-pcie_8gpu-128vcpu-1600gb",
      "gpu-h200-sxm_8gpu-128vcpu-1600gb"
    ], var.slurm_worker_preset)
    error_message = "Invalid worker preset. Must be a valid Nebius GPU preset."
  }
}

# =============================================================================
# Storage Configuration
# =============================================================================

variable "jail_size" {
  description = "Size of shared filesystem (jail) in GB"
  type        = number
  default     = 2048  # 2TB as per PoC requirements
  
  validation {
    condition     = var.jail_size >= 100 && var.jail_size <= 10240
    error_message = "Jail size must be between 100GB and 10TB."
  }
}

variable "network_disk_size" {
  description = "Size of network disk per node in GB"
  type        = number
  default     = 2048  # 2TB as per PoC requirements
  
  validation {
    condition     = var.network_disk_size >= 100 && var.network_disk_size <= 10240
    error_message = "Network disk size must be between 100GB and 10TB."
  }
}

# =============================================================================
# Network Configuration
# =============================================================================

variable "vpc_subnet_cidr" {
  description = "CIDR block for the VPC subnet"
  type        = string
  default     = "10.128.0.0/16"
}

variable "enable_infiniband" {
  description = "Enable InfiniBand networking for GPU cluster"
  type        = bool
  default     = true
}

# =============================================================================
# Optional: Monitoring and Observability
# =============================================================================

variable "enable_monitoring" {
  description = "Enable Prometheus/Grafana monitoring stack"
  type        = bool
  default     = true
}

variable "enable_wandb" {
  description = "Enable Weights & Biases integration"
  type        = bool
  default     = false
}

variable "wandb_api_key" {
  description = "Weights & Biases API key (if enabled)"
  type        = string
  default     = ""
  sensitive   = true
}

# =============================================================================
# Tags/Labels
# =============================================================================

variable "environment" {
  description = "Environment name (poc, dev, staging, prod)"
  type        = string
  default     = "poc"
}

variable "additional_labels" {
  description = "Additional labels to apply to resources"
  type        = map(string)
  default     = {}
}
