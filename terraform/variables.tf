variable "name_prefix" {
  description = "Prefix for every resource name. Change it and all names follow."
  type        = string
  default     = "docintel"
}

variable "location" {
  description = "Needs GPU quota for the training VM family and gpt-4.1-mini availability. eastus has both."
  type        = string
  default     = "eastus"
}

# ------------------------------------------------------------------ compute
# The cluster NAME is fixed so training/job.yml never has to change. Only the
# SIZE is variable: with no Azure ML GPU quota, set training_vm_size to a CPU
# SKU such as Standard_DS3_v2. train.py is torch-only and runs on CPU - the
# 77-second job becomes roughly 40 minutes, and nothing else changes.
variable "training_compute_name" {
  description = "Cluster name. training/job.yml refers to this, so keep it stable."
  type        = string
  default     = "gpu-t4"
}

variable "training_vm_size" {
  description = "One T4 (16 GB). Use a CPU SKU instead if you have no Azure ML GPU quota."
  type        = string
  default     = "Standard_NC4as_T4_v3"
}

# ---------------------------------------------------------------- the agent
variable "agent_model" {
  description = "Azure OpenAI model that runs the agent conversation and tool-calling."
  type        = string
  default     = "gpt-4.1-mini"
}

variable "agent_model_version" {
  type    = string
  default = "2025-04-14"
}

variable "agent_model_capacity" {
  description = "Thousands of tokens per minute. 10 rate-limits a live demo as soon as two people ask at once."
  type        = number
  default     = 150
}

variable "project_name" {
  description = "Foundry project that holds the employee agent. Created by this stack."
  type        = string
  default     = "employee"
}

# ------------------------------------------------------------------- inputs
# The ONLY dependency this stack has on another repository: the storage
# account that Azure-Document-Ingestion created. Read them from that repo's
#   terraform output storage_account
#   terraform output resource_group
variable "ingest_storage_account_name" {
  description = "Storage account created by Azure-Document-Ingestion, holding the curated container."
  type        = string
}

variable "ingest_resource_group_name" {
  description = "Resource group of that storage account."
  type        = string
}

variable "ingest_container" {
  description = "Container with the OCR output and the closed-book datasets."
  type        = string
  default     = "curated"
}

variable "tags" {
  type    = map(string)
  default = { project = "employee-pretraining", managed_by = "terraform" }
}
