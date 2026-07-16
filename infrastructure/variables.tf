# ---- AWS Configuration ----

variable "aws_region" {
  description = "AWS region to deploy in"
  type        = string
  default     = "us-east-1"
}

# ---- Compute Configuration ----

variable "instance_type" {
  description = "Primary EC2 instance type (used for resource tags)"
  type        = string
  default     = "g4dn.2xlarge"
}

variable "instance_types" {
  description = "Ordered list of GPU instance types the fleet may use. Fleet picks best available with price-capacity-optimized strategy."
  type        = list(string)
  default     = ["g5.xlarge", "g4dn.2xlarge"]
  # g5.xlarge   = 1x A10G GPU, 4 vCPUs, 16GB RAM, ~$0.50-0.80/hr spot
  # g4dn.2xlarge = 1x T4 GPU,  8 vCPUs, 32GB RAM, ~$0.30-0.40/hr spot
}

variable "spot_max_price" {
  description = "Maximum hourly price for spot instance (USD)"
  type        = string
  default     = "0.50"
}

# ---- SSH Configuration ----

variable "key_pair_name" {
  description = "Name of existing AWS key pair for SSH access"
  type        = string
}

# ---- Docker Image ----

variable "docker_image" {
  description = "Docker image to pull and run on the instance"
  type        = string
  default     = "mobilenetv2-ssd:latest"
}

# --- ETL Docker Image ----
variable "etl_docker_image" {
  description = "Docker image for the ETL Ray worker"
  type        = string
  default     = "mobilenetv2-ssd-etl:latest"
}

# --- PostgreSQL Database URL for ETL ---
variable "database_url" {
  description = "PostgreSQL connection URL for ETL metadata"
  type        = string
  sensitive   = true
}

# ---- Experiment Configuration ----

variable "experiment_config" {
  description = "Path to experiment config inside the container"
  type        = string
  default     = "../configs/experiments/exp001_baseline.yaml"
}

variable "git_commit" {
  description = "Git commit hash for fingerprinting"
  type        = string
  default     = ""
}

# ---- S3 Configuration ----

variable "s3_checkpoint_bucket" {
  description = "S3 bucket for mid-training checkpoints (without s3:// prefix)"
  type        = string
  default     = "akhilesh-ml-checkpoints"
}

variable "s3_artifacts_bucket" {
  description = "S3 bucket for final model artifacts — saved models, weights, exports (without s3:// prefix)"
  type        = string
  default     = "akhilesh-ml-artifacts"
}

variable "s3_dataset_bucket" {
  description = "S3 bucket for datasets"
  type        = string
  default     = "akhilesh-ml-datasets"
}

variable "experiment_bucket" {
  description = "S3 bucket holding the config library and experiment configs"
  type        = string
  default     = "akhilesh-ml-experiments"
}

# ---- DynamoDB Configuration ----

variable "dynamodb_table_name"{
  description = "Name of the DynamoDB experiment ledger name"
  type = string
  default = "ml-experiment-ledger"
}

# ---- Network Configuration ----

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH (your IP). Use 'x.x.x.x/32' for single IP"
  type        = string
  default     = "0.0.0.0/0"  # WARNING: Open to all. Restrict in production!
}

variable "use_tfrecords" {
  description = "Download TFRecords shards instead of raw dataset"
  type        = bool
  default     = false
}

variable "dataset_name" {
  description = "Dataset folder name in S3 and on disk (e.g. VOCdevkit, VisDrone)"
  type        = string
  default     = "VOCdevkit"
}

