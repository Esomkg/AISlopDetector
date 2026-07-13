output "resource_group_name" {
  value = azurerm_resource_group.aislop.name
}

output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.aislop.name
}

output "acr_login_server" {
  value = azurerm_container_registry.aislop.login_server
}

output "acr_admin_username" {
  value     = azurerm_container_registry.aislop.admin_username
  sensitive = true
}

output "acr_admin_password" {
  value     = azurerm_container_registry.aislop.admin_password
  sensitive = true
}

output "storage_account_name" {
  value = azurerm_storage_account.aislop.name
}

output "storage_account_key" {
  value     = azurerm_storage_account.aislop.primary_access_key
  sensitive = true
}

output "redis_hostname" {
  value = azurerm_redis_cache.aislop.hostname
}

output "redis_primary_key" {
  value     = azurerm_redis_cache.aislop.primary_access_key
  sensitive = true
}

output "kube_config" {
  value     = azurerm_kubernetes_cluster.aislop.kube_config_raw
  sensitive = true
}

output "ml_workspace_name" {
  value = azurerm_machine_learning_workspace.aislop.name
}

output "ml_compute_cluster_name" {
  value = azurerm_machine_learning_compute_cluster.gpu.name
}
