#!/bin/bash
set -e

echo "=== TFM-SLM Build & Deploy on EC2 ==="

# Configuration
AWS_REGION="eu-south-2"
S3_BUCKET="tfm-slm-code"
S3_KEY="tfm-slm-code.zip"
ECR_REPO="tfm-slm"
IMAGE_TAG="latest"
MODE="${MODE:-train}"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"

echo "IMAGE_URI: $IMAGE_URI"
echo "MODE: $MODE"

# Training mode: build, push to ECR, train
if [ "$MODE" = "train" ]; then
    echo "=== TRAINING MODE ==="

    # Code already extracted by EC2 UserData, current directory is /home/ec2-user/tfm-slm
    echo "Working directory: $(pwd)"
    echo "Files in current directory:"
    ls -la

    # Stage 1: Build Docker image with GPU
    echo "=== Building Docker image with GPU ==="
    docker build -t "$IMAGE_URI" .

    # Stage 2: Push to ECR
    echo "=== Pushing to ECR ==="
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    docker push "$IMAGE_URI"
    echo "Image pushed to ECR: $IMAGE_URI"

    # Stage 3: Run training
    echo "=== Running training ==="
    docker run --gpus all \
      -e MODE="train" \
      -v ~/.aws/credentials:/root/.aws/credentials:ro \
      "$IMAGE_URI"

# Inference mode: build fresh image, push to ECR, run inference
elif [ "$MODE" = "inference" ]; then
    echo "=== INFERENCE MODE ==="

    # Stage 1: Build Docker image (ensures latest code changes are included)
    echo "=== Building Docker image ==="
    docker build -t "$IMAGE_URI" .

    # Stage 2: Push to ECR
    echo "=== Pushing to ECR ==="
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    docker push "$IMAGE_URI"
    echo "Image pushed to ECR: $IMAGE_URI"

    # Stage 3: Run inference
    echo "=== Running inference ==="
    docker run --gpus all -it \
      -e MODE="inference" \
      -v ~/.aws/credentials:/root/.aws/credentials:ro \
      "$IMAGE_URI"

# Benchmark mode: build image, push to ECR, run benchmarking
elif [ "$MODE" = "benchmark" ]; then
    echo "=== BENCHMARK MODE ==="

    # Stage 1: Build Docker image
    echo "=== Building Docker image ==="
    docker build -t "$IMAGE_URI" .

    # Stage 2: Push to ECR
    echo "=== Pushing to ECR ==="
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    docker push "$IMAGE_URI"
    echo "Image pushed to ECR: $IMAGE_URI"

    # Stage 3: Run benchmarking (evaluates checkpoint on 300K samples)
    echo "=== Running benchmarking ==="
    docker run --gpus all \
      -e MODE="benchmark" \
      -v ~/.aws/credentials:/root/.aws/credentials:ro \
      "$IMAGE_URI" \
      uv run tfm-slm-benchmark

else
    echo "ERROR: Unknown mode '$MODE'. Valid modes: train, inference, benchmark"
    exit 1
fi

echo "=== Complete ==="
