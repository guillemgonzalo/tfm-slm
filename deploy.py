#!/usr/bin/env python3
"""
Deployment script: Upload code to S3 for EC2 build with GPU.

Flow:
  1. deploy.py → archives code, uploads to S3
  2. terraform apply → creates EC2 instance
  3. EC2 UserData → downloads code, builds Docker with GPU (flash-attn), trains

boto3 automatically reads AWS credentials from ~/.aws/credentials.
No environment variables or exports needed.

Ensure ~/.aws/credentials exists with:
  [default]
  aws_access_key_id = YOUR_KEY
  aws_secret_access_key = YOUR_SECRET
"""

import logging
import zipfile
from pathlib import Path

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = "eu-south-2"
S3_CODE_BUCKET = "tfm-slm-code"
S3_BENCHMARK_BUCKET = "tfm-slm-benchmarks"
ECR_REPO = "tfm-slm"
DOCKER_IMAGE_TAG = "latest"
CODE_ZIP = "tfm-slm-code.zip"


def archive_code(output_file: str = CODE_ZIP) -> None:
    """Archive project code to zip file (minimal build files only)."""
    # Delete all existing ZIP files to avoid nested ZIPs
    for zip_file in Path(".").glob("*.zip"):
        logger.info(f"Removing existing {zip_file}...")
        zip_file.unlink()

    logger.info(f"Archiving code to {output_file}...")

    # Include only essential files for Docker build
    include_files = {
        "pyproject.toml",
        "uv.lock",
        "Dockerfile",
        "entrypoint.sh",
        "build-and-train.sh",
    }

    # Include app/ directory
    include_dirs = {"app"}

    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add specific files
        for file in include_files:
            path = Path(file)
            if path.exists() and path.is_file():
                zipf.write(path)
                logger.debug(f"Added: {path}")

        # Add app/ directory recursively, skip __pycache__
        for path in Path("app").rglob("*"):
            if path.is_file():
                if "__pycache__" not in path.parts:
                    zipf.write(path)
                    logger.debug(f"Added: {path}")

    logger.info(f"Code archived: {output_file}")


def upload_to_s3(file_path: str, bucket: str, key: str) -> None:
    """Upload file to S3."""
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    try:
        logger.info(f"Uploading {file_path} to s3://{bucket}/{key}...")
        s3_client.upload_file(file_path, bucket, key)
        logger.info(f"Uploaded to S3: s3://{bucket}/{key}")
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise


def ensure_ecr_repo() -> None:
    """Ensure ECR repository exists."""
    ecr_client = boto3.client("ecr", region_name=AWS_REGION)

    try:
        logger.info(f"Checking ECR repository: {ECR_REPO}...")
        ecr_client.describe_repositories(repositoryNames=[ECR_REPO])
        logger.info(f"ECR repository exists: {ECR_REPO}")
    except ecr_client.exceptions.RepositoryNotFoundException:
        logger.info(f"Creating ECR repository: {ECR_REPO}...")
        ecr_client.create_repository(repositoryName=ECR_REPO)
        logger.info(f"ECR repository created: {ECR_REPO}")


def ensure_benchmark_bucket() -> None:
    """Ensure S3 benchmark bucket exists in eu-south-2."""
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    try:
        logger.info(f"Checking S3 benchmark bucket: {S3_BENCHMARK_BUCKET}...")
        s3_client.head_bucket(Bucket=S3_BENCHMARK_BUCKET)
        logger.info(f"Benchmark bucket exists: {S3_BENCHMARK_BUCKET}")
    except s3_client.exceptions.NoSuchBucket:
        logger.info(f"Creating S3 benchmark bucket: {S3_BENCHMARK_BUCKET}...")
        s3_client.create_bucket(
            Bucket=S3_BENCHMARK_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
        )
        logger.info(f"Benchmark bucket created: {S3_BENCHMARK_BUCKET}")
    except Exception as e:
        logger.warning(f"Could not check benchmark bucket: {e}")


def main():
    """Main deployment flow."""
    logger.info("=" * 60)
    logger.info("TFM-SLM Deployment Pipeline")
    logger.info("=" * 60)

    # Ask user for deployment mode
    print("\nSelect deployment mode:")
    print("  1. Training (download + process + train)")
    print("  2. Inference (load checkpoint + chat)")
    print("  3. Benchmarking (evaluate checkpoint on 300K samples)")
    choice = input("\nEnter choice (1, 2 or 3): ").strip()

    if choice == "1":
        mode = "train"
        logger.info("Mode: TRAINING")
    elif choice == "2":
        mode = "inference"
        logger.info("Mode: INFERENCE")
    elif choice == "3":
        mode = "benchmark"
        logger.info("Mode: BENCHMARKING")
    else:
        logger.error("Invalid choice. Exiting.")
        return

    # Save mode to terraform.tfvars (replace existing)
    tfvars_path = "app/terraform/terraform.tfvars"
    with open(tfvars_path, "r") as f:
        lines = f.readlines()
    lines = [l for l in lines if not l.strip().startswith("deployment_mode")]
    with open(tfvars_path, "w") as f:
        f.writelines(lines)
        f.write(f'deployment_mode = "{mode}"\n')
    logger.info(f"Saved deployment_mode = {mode} to terraform.tfvars")

    try:
        # 1. Archive code
        archive_code()

        # 2. Upload to S3
        upload_to_s3(CODE_ZIP, S3_CODE_BUCKET, CODE_ZIP)

        # 3. Ensure ECR repo exists
        ensure_ecr_repo()

        # 4. Ensure benchmark bucket exists
        ensure_benchmark_bucket()

        logger.info("=" * 60)
        logger.info("Code uploaded to S3!")
        logger.info("=" * 60)

        # Clean up local ZIP after successful upload
        zip_path = Path(CODE_ZIP)
        if zip_path.exists():
            logger.info(f"Cleaning up local {CODE_ZIP}...")
            zip_path.unlink()

        next_steps = (
            f"Next steps:\n"
            f"  1. cd app/terraform\n"
            f"  2. terraform apply -auto-approve -lock=false\n"
            f"  3. EC2 will:\n"
            f"     - Download code from S3\n"
            f"     - Build Docker image with GPU (flash-attn compiles)\n"
            f"     - Push image to ECR\n"
        )

        if mode == "train":
            next_steps += (
                f"     - Run training (15 epochs ~2-4 hours)\n"
            )
        elif mode == "inference":
            next_steps += (
                f"     - Run interactive chat inference\n"
            )
        elif mode == "benchmark":
            next_steps += (
                f"     - Run benchmarking (evaluate 300K samples, save to S3)\n"
            )

        next_steps += (
            f"  4. Monitor with: ssh -i \"tfm-slm.pem\" ec2-user@<IP> tail -f /var/log/user-data.log"
        )

        logger.info(next_steps)

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()
