variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
  sensitive   = true
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default     = "aislop-detector-rg"
}

variable "cluster_name" {
  description = "AKS cluster name"
  type        = string
  default     = "aislop-aks"
}

variable "acr_name" {
  description = "Azure Container Registry name (alphanumeric only)"
  type        = string
  default     = "aislopacr"
}

variable "storage_account_name" {
  description = "Storage account name (alphanumeric only)"
  type        = string
  default     = "aislopstorage"
}

variable "redis_name" {
  description = "Redis cache name"
  type        = string
  default     = "aislop-redis"
}

variable "node_count" {
  description = "Default node pool count"
  type        = number
  default     = 2
}

variable "node_vm_size" {
  description = "Default node VM size"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "gpu_vm_size" {
  description = "GPU node VM size"
  type        = string
  default     = "Standard_NC6s_v3"
}

variable "gpu_node_count" {
  description = "GPU node pool initial count"
  type        = number
  default     = 0
}

variable "gpu_max_nodes" {
  description = "GPU node pool max autoscale count"
  type        = number
  default     = 3
}

variable "ml_workspace_name" {
  description = "Azure ML Workspace name"
  type        = string
  default     = "aislop-ml"
}

variable "ml_gpu_vm_size" {
  description = "Azure ML GPU compute VM size"
  type        = string
  default     = "Standard_NC6s_v3"
}

variable "ml_gpu_max_nodes" {
  description = "Azure ML GPU compute max autoscale nodes"
  type        = number
  default     = 4
}

variable "use_spot_vms" {
  description = "Use spot VMs for Azure ML compute (~80% cheaper)"
  type        = bool
  default     = true
}
