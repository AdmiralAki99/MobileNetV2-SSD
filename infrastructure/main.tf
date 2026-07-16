# The "terraform" block configures Terraform itself
terraform {
  # Which providers (cloud APIs) we need
  required_providers {
    aws = {
      source  = "hashicorp/aws"   # Official AWS provider by HashiCorp
      version = "~> 5.0"          # Use version 5.x (~ means compatible)
    }
  }

  required_version = ">= 1.5.0"  # Minimum Terraform version
}

provider "aws" {
  region = var.aws_region  # Which AWS region to create resources in
}
