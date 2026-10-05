terraform {
  required_version = ">= 1.9"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    # The Foundry project (Microsoft.CognitiveServices/accounts/projects) has no
    # azurerm resource yet, so it is created through the generic ARM provider
    # instead of a manual `az rest` call.
    azapi = {
      source  = "Azure/azapi"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "azapi" {}

provider "azurerm" {
  storage_use_azuread = true
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    key_vault {
      purge_soft_delete_on_destroy = true
    }
    cognitive_account {
      purge_soft_delete_on_destroy = true
    }
  }
}
