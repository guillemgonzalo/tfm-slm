variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-south-2"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "tfm-slm"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g7e.2xlarge"
}

variable "ssh_key_name" {
  description = "Name of the SSH key pair"
  type        = string
  default     = "tfm-slm"
}

variable "docker_image_tag" {
  description = "Tag of the Docker image to deploy"
  type        = string
  default     = "latest"
}

variable "spot_price" {
  description = "Maximum price for Spot instance"
  type        = string
  default     = "1.50"
}

variable "deployment_mode" {
  description = "Deployment mode: train, inference, or benchmark"
  type        = string
  default     = "train"
  validation {
    condition     = contains(["train", "inference", "benchmark"], var.deployment_mode)
    error_message = "deployment_mode must be 'train', 'inference', or 'benchmark'"
  }
}

variable "s3_code_bucket" {
  description = "S3 bucket where code is uploaded"
  type        = string
  default     = "tfm-slm-code"
}

variable "code_zip_key" {
  description = "S3 key for code ZIP file"
  type        = string
  default     = "tfm-slm-code.zip"
}
