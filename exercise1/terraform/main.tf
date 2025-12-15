# main.tf - Nebius Soperator Infrastructure for LLM Fine-tuning PoC

terraform {
  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = ">= 0.5.0"
    }
  }
  required_version = ">= 1.5.0"
}

provider "nebius" {
  # Credentials configured via nebius init or environment variables
}

# Local variables for configuration
locals {
  cluster_name = var.slurm_cluster_name
  region       = var.region
  zone         = "${var.region}-a"
  
  # Labels for resource organization
  labels = {
    environment = "poc"
    project     = "llm-finetune"
    customer    = "poc-customer"
  }
}

# =============================================================================
# VPC Network Configuration
# =============================================================================

resource "nebius_vpc_network" "training_network" {
  name      = "${local.cluster_name}-network"
  folder_id = var.folder_id
  labels    = local.labels
}

resource "nebius_vpc_subnet" "training_subnet" {
  name           = "${local.cluster_name}-subnet"
  zone           = local.zone
  network_id     = nebius_vpc_network.training_network.id
  v4_cidr_blocks = [var.vpc_subnet_cidr]
  folder_id      = var.folder_id
  labels         = local.labels
}

# =============================================================================
# GPU Cluster for InfiniBand Networking
# =============================================================================

resource "nebius_compute_gpu_cluster" "training_cluster" {
  name               = "${local.cluster_name}-gpu-cluster"
  folder_id          = var.folder_id
  zone               = local.zone
  interconnect_type  = "infiniband"
  
  labels = local.labels
}

# =============================================================================
# Managed Kubernetes Cluster
# =============================================================================

resource "nebius_mk8s_cluster" "k8s_cluster" {
  name        = "${local.cluster_name}-k8s"
  folder_id   = var.folder_id
  
  network_id  = nebius_vpc_network.training_network.id
  
  master {
    version = "1.29"
    zonal {
      zone      = local.zone
      subnet_id = nebius_vpc_subnet.training_subnet.id
    }
    
    maintenance_policy {
      auto_upgrade = false
      maintenance_window {
        day        = "sunday"
        start_time = "03:00"
        duration   = "3h"
      }
    }
  }
  
  labels = local.labels
}

# =============================================================================
# Kubernetes Node Groups
# =============================================================================

# System node group for control plane components
resource "nebius_mk8s_node_group" "system_nodes" {
  cluster_id = nebius_mk8s_cluster.k8s_cluster.id
  name       = "system-nodes"
  
  instance_template {
    platform_id = "standard-v3"
    
    resources {
      cores  = 8
      memory = 32
    }
    
    boot_disk {
      type = "network-ssd"
      size = 100
    }
    
    network_interface {
      subnet_ids = [nebius_vpc_subnet.training_subnet.id]
      nat        = true
    }
  }
  
  scale_policy {
    auto_scale {
      min     = 1
      max     = 3
      initial = 2
    }
  }
  
  allocation_policy {
    location {
      zone = local.zone
    }
  }
  
  labels = local.labels
}

# GPU worker node group (2 nodes × 8 H100 GPUs = 16 GPUs)
resource "nebius_mk8s_node_group" "gpu_workers" {
  cluster_id = nebius_mk8s_cluster.k8s_cluster.id
  name       = "gpu-workers"
  
  instance_template {
    platform_id = "gpu-h100-sxm"
    
    resources {
      cores         = 128
      memory        = 1600
      gpus          = 8
      gpu_cluster_id = nebius_compute_gpu_cluster.training_cluster.id
    }
    
    boot_disk {
      type = "network-ssd-nonreplicated"
      size = 500
    }
    
    # Additional network disk for data
    secondary_disk {
      device_name = "data-disk"
      disk_spec {
        type = "network-ssd-nonreplicated"
        size = var.network_disk_size
      }
    }
    
    network_interface {
      subnet_ids = [nebius_vpc_subnet.training_subnet.id]
      nat        = false  # Access via login node
    }
  }
  
  scale_policy {
    fixed_scale {
      size = var.slurm_worker_count  # 2 nodes for PoC
    }
  }
  
  allocation_policy {
    location {
      zone = local.zone
    }
  }
  
  node_labels = {
    "nvidia.com/gpu"     = "present"
    "node-role"          = "gpu-worker"
  }
  
  node_taints {
    key    = "nvidia.com/gpu"
    value  = "present"
    effect = "NO_SCHEDULE"
  }
  
  labels = local.labels
}

# =============================================================================
# Shared File Storage (2TB for jail/shared filesystem)
# =============================================================================

resource "nebius_compute_filesystem" "shared_storage" {
  name      = "${local.cluster_name}-shared"
  folder_id = var.folder_id
  zone      = local.zone
  
  type = "network-ssd"
  size = var.jail_size  # 2TB
  
  labels = local.labels
}

# =============================================================================
# Soperator Helm Release
# =============================================================================

# Note: This requires the Kubernetes provider and Helm provider to be configured
# after the cluster is created. In practice, you would apply this separately
# or use a two-stage deployment.

# provider "kubernetes" {
#   host                   = nebius_mk8s_cluster.k8s_cluster.master[0].external_v4_endpoint
#   cluster_ca_certificate = base64decode(nebius_mk8s_cluster.k8s_cluster.master[0].cluster_ca_certificate)
#   token                  = data.nebius_iam_token.k8s_token.iam_token
# }

# provider "helm" {
#   kubernetes {
#     host                   = nebius_mk8s_cluster.k8s_cluster.master[0].external_v4_endpoint
#     cluster_ca_certificate = base64decode(nebius_mk8s_cluster.k8s_cluster.master[0].cluster_ca_certificate)
#     token                  = data.nebius_iam_token.k8s_token.iam_token
#   }
# }

# resource "helm_release" "soperator" {
#   name       = "soperator"
#   repository = "https://nebius.github.io/soperator"
#   chart      = "soperator"
#   namespace  = "soperator-system"
#   
#   create_namespace = true
#   
#   values = [
#     yamlencode({
#       workers = {
#         count = var.slurm_worker_count
#       }
#       storage = {
#         jailSize = "${var.jail_size}Gi"
#       }
#     })
#   ]
# }

# =============================================================================
# Outputs
# =============================================================================

output "k8s_cluster_id" {
  description = "Kubernetes cluster ID"
  value       = nebius_mk8s_cluster.k8s_cluster.id
}

output "k8s_cluster_endpoint" {
  description = "Kubernetes API endpoint"
  value       = nebius_mk8s_cluster.k8s_cluster.master[0].external_v4_endpoint
  sensitive   = true
}

output "gpu_cluster_id" {
  description = "GPU cluster ID for InfiniBand"
  value       = nebius_compute_gpu_cluster.training_cluster.id
}

output "shared_filesystem_id" {
  description = "Shared filesystem ID"
  value       = nebius_compute_filesystem.shared_storage.id
}

output "network_id" {
  description = "VPC Network ID"
  value       = nebius_vpc_network.training_network.id
}

output "subnet_id" {
  description = "Subnet ID"
  value       = nebius_vpc_subnet.training_subnet.id
}

output "cluster_info" {
  description = "Cluster connection information"
  value = {
    cluster_name = local.cluster_name
    region       = local.region
    zone         = local.zone
    gpu_count    = var.slurm_worker_count * 8
    worker_nodes = var.slurm_worker_count
  }
}
