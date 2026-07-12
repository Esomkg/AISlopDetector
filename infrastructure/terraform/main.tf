terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

# --- Resource Group ---
resource "azurerm_resource_group" "aislop" {
  name     = var.resource_group_name
  location = var.location
}

# --- Azure Container Registry ---
resource "azurerm_container_registry" "aislop" {
  name                = replace(var.acr_name, "-", "")
  resource_group_name = azurerm_resource_group.aislop.name
  location            = azurerm_resource_group.aislop.location
  sku                 = "Basic"
  admin_enabled       = true
}

# --- Blob Storage for DVC / datasets ---
resource "azurerm_storage_account" "aislop" {
  name                     = replace(var.storage_account_name, "-", "")
  resource_group_name      = azurerm_resource_group.aislop.name
  location                 = azurerm_resource_group.aislop.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_container" "datasets" {
  name                  = "datasets"
  storage_account_name  = azurerm_storage_account.aislop.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.aislop.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "embeddings" {
  name                  = "embeddings"
  storage_account_name  = azurerm_storage_account.aislop.name
  container_access_type = "private"
}

# --- Azure Cache for Redis (Feast online store) ---
resource "azurerm_redis_cache" "aislop" {
  name                = var.redis_name
  resource_group_name = azurerm_resource_group.aislop.name
  location            = azurerm_resource_group.aislop.location
  capacity            = 0
  family              = "C"
  sku_name            = "Basic"
  minimum_tls_version = "1.2"
}

# --- AKS Cluster ---
resource "azurerm_virtual_network" "aislop" {
  name                = "${var.cluster_name}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.aislop.location
  resource_group_name = azurerm_resource_group.aislop.name
}

resource "azurerm_subnet" "aks" {
  name                 = "${var.cluster_name}-subnet"
  resource_group_name  = azurerm_resource_group.aislop.name
  virtual_network_name = azurerm_virtual_network.aislop.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_kubernetes_cluster" "aislop" {
  name                = var.cluster_name
  location            = azurerm_resource_group.aislop.location
  resource_group_name = azurerm_resource_group.aislop.name
  dns_prefix          = var.cluster_name

  default_node_pool {
    name       = "default"
    node_count = var.node_count
    vm_size    = var.node_vm_size
    vnet_subnet_id = azurerm_subnet.aks.id
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "azure"
  }

  depends_on = [azurerm_subnet.aks]
}

# --- GPU Node Pool for model training ---
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpunodes"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aislop.id
  vm_size               = var.gpu_vm_size
  node_count            = var.gpu_node_count
  node_taints           = ["nvidia.com/gpu=true:NoSchedule"]
  node_labels = {
    "accelerator" = "nvidia-gpu"
  }
  enable_auto_scaling = true
  min_count           = 0
  max_count           = var.gpu_max_nodes
}

# --- ACR role assignment for AKS ---
resource "azurerm_role_assignment" "aks_acr_pull" {
  principal_id                     = azurerm_kubernetes_cluster.aislop.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.aislop.id
  skip_service_principal_aad_check = true
}
