# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

```bash
# Install dependencies
uv sync

# Run the complete training pipeline (download → process → train)
uv run tfm-slm

# Run inference with trained checkpoint (separate session)
uv run tfm-slm-inference

# Linting and formatting (ruff)
ruff check .
ruff format .

# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=app

# Type checking
ty check

# Pre-commit checks (all hooks)
pre-commit run

# Interactive chat inference (requires checkpoint)
uv run python -m app.chat.chat

# Build Docker image
docker build -t tfm-slm:latest .

# Deployment to AWS (local, no GitHub, build in AWS with CodeBuild)
echo "1" | python3 deploy.py        # Select train/inference, upload to S3
cd app/terraform && terraform apply # Provision EC2 with selected mode
```

## Architecture Overview

### Core Design: HybridBlock

The model is built around the **HybridBlock** component (`app/model/architecture.py`), which is the fundamental building block of the model. Each block performs:

1. **Pre-LayerNorm + Multi-Head Attention (MHA)**: Captures global context and long-range dependencies across up to 1024 tokens using 12 attention heads.
2. **GRU Layer**: Sequential refinement mechanism that acts as a local pattern filter after attention, reducing the need for additional attention layers.
3. **FFN (Feed-Forward Network)**: MLP with expansion factor 4 and GELU activation for feature processing.

The model stacks **12 HybridBlocks** (~124-167M parameters) and uses **weight tying** between token embeddings and the output head to reduce VRAM by 18%.

### Pipeline Architecture: Four Orchestrated Services

The main entry point (`app/main.py`) orchestrates four independent services following SOLID principles:

1. **DatasetDownloader** (`app/dataset/downloader.py`)
   - Downloads raw data from HuggingFace (OpenAssistant, ShareGPT, Alpaca, UltraChat, The Stack)
   - Stores raw files locally

2. **DatasetProcessor** (`app/dataset/processor.py`)
   - Mixes datasets with configured ratios (50% conversational, 50% instruction, specialty subsets)
   - Pre-tokenizes using GPT-2 tokenizer (50,257 tokens)
   - Outputs Apache Arrow format (.arrow) for memory-mapped training

3. **TrainingService** (`app/training/trainer.py`)
   - Loads HybridModel from HybridConfig
   - Implements torch.compile for kernel fusion and optimization
   - Supports causal mask caching to reduce VRAM pressure
   - Checkpoints automatically to S3 (`tfm-slm-checkpoints` bucket) after each epoch

4. **ChatService** (`app/chat/chat.py`)
   - Interactive chat inference after training completes
   - Auto-downloads checkpoint from S3 if not available locally
   - Loads tokenizer and model, runs chat loop

### Configuration Management

**app/config.py** uses Pydantic's `BaseSettings` to manage:
- Dataset directory paths
- S3 bucket names for checkpoints
- Hardware specs (GPU VRAM capacity: 96GB for RTX PRO 6000 Blackwell)
- HuggingFace trust settings

Settings can be overridden via `.env` file or environment variables.

## Key Implementation Details

### Model Configuration
- **Hidden Size**: 768
- **Number of Layers**: 12 HybridBlocks
- **Attention Heads**: 12
- **Intermediate FFN Size**: 3072 (4x expansion)
- **Max Context Length**: 1024 tokens
- **Dropout**: 0.1

### Training Hyperparameters (Optimized for RTX PRO 6000 Blackwell)
- **Physical Batch Size**: 64
- **Gradient Accumulation Steps**: 6
- **Effective Batch Size**: 384
- **Epochs**: 15 (configurable in main.py)
- **Precision**: bfloat16 with TF32 matmul acceleration
- **Learning Rate**: 5e-5 with cosine annealing + warmup (10% of total steps)
- **Optimizer**: AdamW with weight_decay=0.01

### Optimizations (May 2026)

#### Flash Attention (2-3x speedup on NVIDIA GPU)
- I/O-aware implementation reduces memory O(n²) → O(n)
- Pre-built wheel: `flash-attn==2.8.3+cu12torch2.11` installed via pip in Docker (AWS/Linux only)
- Code only executes in AWS (Linux/CUDA). Mac is for editing only.
- Implementation: `app/model/architecture.py:FlashAttention`

#### Learning Rate Scheduler
- Cosine annealing with warmup improves convergence by ~15% epochs
- Warmup phase (10% of steps): LR increases from 0 to target
- Annealing phase: LR decreases cosine from target to target/100
- Implementation: `app/training/trainer.py` uses `CosineAnnealingWarmRestarts`

### Data Mixing Strategy
- **50% Conversational**: OpenAssistant + UltraChat (dialogue naturalness)
- **50% Instruction**: Alpaca + ShareGPT (instruction-following)
- **Specialty**: The Stack (YAML subset) for infrastructure/DevOps syntax

## Testing & CI/CD

### Local Testing
- Test files in `tests/` mirror the `app/` structure
- Use `pytest` with coverage tracking
- Pre-commit hooks configured in `.pre-commit-config.yaml`

### Deployment (Local, no GitHub)
- **Code Upload + Docker Build**: `python deploy.py` archives code to S3, builds Docker image, pushes to ECR
  - Reads AWS credentials from `~/.aws/credentials` automatically
  - Creates ECR repository if not exists
  - Excludes large files and development directories
- **Infrastructure Provisioning**: `cd app/terraform && terraform apply` provisions EC2 On-Demand instances, S3 buckets, security groups
  - Reads configuration from `terraform.tfvars` (local, not versioned)
  - Uses Terraform remote state from S3 for lock/sync

### Docker Requirements
The runtime image must include:
- `gcc`, `g++`, `python3-dev` (required for `torch.compile` and Triton kernel compilation)
- All Python dependencies from `pyproject.toml`

## Development Notes

### Linting & Code Standards
- **Target Python**: 3.13+ with built-in generics (e.g., `list[int]`)
- **Ruff Config**: Line length 88, select E/F/I/B/S/UP, target Python 3.13
- Tests exempt from S101 (assert statements allowed)

### Checkpoint Management
- Checkpoints are automatically synced to S3 (`tfm-slm-checkpoints` bucket) using raw `state_dict` (no torch.compile prefixes) for maximum compatibility
- After training completes, interactive chat session starts automatically (ChatService in `app/chat/`)
- ChatService auto-downloads checkpoint from S3 if not available locally

### Performance Optimizations
- **Flash Attention**: O(n) memory, 2-3x faster on modern NVIDIA GPUs. Enabled in `app/model/architecture.py:FlashAttention`
- **Cosine Annealing LR Scheduler**: 10% warmup + cosine decay improves convergence by ~15% epochs and +1-2% accuracy
- **torch.compile**: Kernel fusion and graph optimization for Blackwell architecture (max-autotune, no CUDAGraphs)
- **bfloat16 + TF32**: Mixed precision training reduces VRAM while maintaining accuracy

### Infrastructure as Code
Terraform configurations in `app/terraform/` manage AWS resources (On-Demand instances, S3 buckets, security groups, IAM roles).

Setup:
1. Copy `terraform.tfvars.example` to `terraform.tfvars` (not versioned)
2. Edit `terraform.tfvars` with region, instance type, key pair name
3. Run `terraform init` (one-time)
4. Run `terraform plan` to review changes
5. Run `terraform apply` to provision infrastructure

No GitHub dependency — Terraform runs locally with AWS credentials from `~/.aws/credentials`.

## Additional Resources

- **README.md**: User-facing documentation and setup instructions
- **report/**: LaTeX thesis chapters (1-7) with detailed methodology and results
