import logging
import os
from pathlib import Path

import boto3
import torch
import torch.optim as optim
from app.config import settings
from app.model.architecture import HybridConfig, HybridModel
from app.utils.analyzer import HybridArchitectureAnalyzer
from botocore.exceptions import ClientError
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optimizations for NVIDIA RTX PRO 6000 Blackwell
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")  # Enables TF32 for faster matmuls
torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Service responsible for the training loop of the SLM,
    highly optimized for NVIDIA RTX PRO 6000 Blackwell.
    """

    def __init__(self, dataset_path: str = ".datasets/mixed_dataset"):
        self.dataset_path = Path(dataset_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s3_client = boto3.client("s3")
        self.bucket_name = settings.checkpoint_bucket

        # Reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        logger.info(f"Using device: {self.device}")

        from app.dataset.tokenizer import TokenizerService
        self.tokenizer = TokenizerService().load()

    def _clip_grad_norm(self, model, max_norm: float = 1.0):
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return total_norm

    def _analyze_component_contributions(self, model):
        gru_norm = 0.0
        attn_norm = 0.0
        mlp_norm = 0.0

        for block in model.blocks:
            for name, param in block.named_parameters():
                if param.grad is not None:
                    if 'gru' in name:
                        gru_norm += param.grad.norm().item() ** 2
                    elif 'attn' in name:
                        attn_norm += param.grad.norm().item() ** 2
                    elif 'mlp' in name:
                        mlp_norm += param.grad.norm().item() ** 2

        gru_norm = gru_norm ** 0.5
        attn_norm = attn_norm ** 0.5
        mlp_norm = mlp_norm ** 0.5

        gru_to_attn_ratio = gru_norm / (attn_norm + 1e-8)

        return {
            "gru_norm": gru_norm,
            "attn_norm": attn_norm,
            "mlp_norm": mlp_norm,
            "gru_to_attn_ratio": gru_to_attn_ratio,
        }

    def _validate(self, model, dataloader, precision, max_batches: int = 200):
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                if n >= max_batches:
                    break
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=precision):
                    outputs = model(input_ids, labels=input_ids)
                    val_loss += outputs["loss"].item()
                n += 1
        model.train()
        return val_loss / max(n, 1)

    def train(
        self,
        epochs: int = 1,
        batch_size: int = 64,
        grad_accum_steps: int = 1,
        lr: float = 5e-5,
        grad_clip_norm: float = 1.0,
        log_metrics_every: int = 100,
        validate_every: int = 500,
    ):
        """
        Trains the hybrid architecture from scratch.
        Leverages bfloat16, TF32, and torch.compile for performance on Blackwell.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. Run processing first."
            )

        # 1. Load Pre-tokenized Dataset
        logger.info(f"Loading pre-tokenized dataset from {self.dataset_path}...")
        tokenized_ds = load_from_disk(str(self.dataset_path))
        tokenized_ds.set_format("torch")

        # 2. Optimized DataLoader
        dataloader = DataLoader(
            tokenized_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=min(8, os.cpu_count() or 1),
            prefetch_factor=4,
        )

        # 3. Model Initialization (Hybrid Transformer-GRU)
        config = HybridConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=1024,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072,
        )

        logger.info("Initializing Hybrid Transformer-GRU Model...")
        model = HybridModel(config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # Learning Rate Scheduler: Warmup + Cosine Annealing
        steps_per_epoch = len(dataloader) // grad_accum_steps
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=total_steps - warmup_steps, eta_min=lr / 100
        )

        # Initialize analyzer for component contribution tracking
        analyzer = HybridArchitectureAnalyzer(model, self.device)

        # 4. Checkpoint Resuming
        checkpoint_dir = Path(".output")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        start_epoch = 0

        def _load_checkpoint(ckpt_path: Path) -> tuple[int, int]:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint["epoch"] + 1
            step = checkpoint.get("global_step", epoch * steps_per_epoch)
            logger.info(f"Resuming from epoch {epoch}, global step {step}")
            return epoch, step

        restored_step = 0
        if checkpoint_path.exists():
            logger.info(f"Checkpoint found at {checkpoint_path}. Resuming...")
            start_epoch, restored_step = _load_checkpoint(checkpoint_path)
        else:
            try:
                logger.info(f"Checking S3 bucket {self.bucket_name} for checkpoints...")
                self.s3_client.download_file(
                    self.bucket_name, "checkpoint.pt", str(checkpoint_path)
                )
                logger.info("Checkpoint downloaded from S3. Resuming...")
                start_epoch, restored_step = _load_checkpoint(checkpoint_path)
            except ClientError:
                logger.info("No checkpoint found in S3. Starting from scratch.")

        # torch.compile fusion for Blackwell architecture
        # Compile AFTER loading to avoid state_dict key mismatches
        logger.info("Compiling model with torch.compile for maximum throughput...")
        model = torch.compile(model, options={"triton.cudagraphs": False, "max_autotune": True})

        # Blackwell supports bfloat16
        precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Using mixed precision: {precision}")

        # GradScaler is only needed for float16. bfloat16 doesn't need scaling.
        scaler = torch.amp.GradScaler("cuda", enabled=(precision == torch.float16))

        msg = f"Training started. Batch: {batch_size}, Accum: {grad_accum_steps}, GradClip: {grad_clip_norm}"
        logger.info(msg)
        model.train()

        global_step = restored_step if restored_step else start_epoch * steps_per_epoch

        for epoch in range(start_epoch, epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for i, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)

                # Autocast to bfloat16/float16
                with torch.amp.autocast("cuda", dtype=precision):
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs["loss"] / grad_accum_steps

                if precision == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % grad_accum_steps == 0:
                    grad_before = self._clip_grad_norm(model._orig_mod if hasattr(model, "_orig_mod") else model, max_norm=grad_clip_norm)

                    if precision == torch.float16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    # Component analysis before zero_grad (grads still available)
                    val_loss = None
                    if (global_step + 1) % log_metrics_every == 0:
                        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                        metrics = self._analyze_component_contributions(raw_model)

                    optimizer.zero_grad(set_to_none=True)

                    if (global_step + 1) % log_metrics_every == 0:
                        logger.info(
                            f"[Step {global_step + 1}] Loss: {loss.item() * grad_accum_steps:.4f} | "
                            f"GRU norm: {metrics['gru_norm']:.4f} | "
                            f"Attn norm: {metrics['attn_norm']:.4f} | "
                            f"MLP norm: {metrics['mlp_norm']:.4f} | "
                            f"GRU/Attn ratio: {metrics['gru_to_attn_ratio']:.3f} | "
                            f"Grad (before/after): {grad_before:.3f}/1.000"
                        )
                        analyzer.log_metrics(
                            step=global_step + 1,
                            loss=loss.item() * grad_accum_steps,
                            grad_norm_before=grad_before,
                            grad_norm_after=1.0,
                            component_metrics=metrics,
                        )

                    # Validation
                    if (global_step + 1) % validate_every == 0:
                        val_loss = self._validate(model, dataloader, precision)
                        logger.info(f"[Step {global_step + 1}] Validation Loss: {val_loss:.4f}")
                        analyzer.log_metrics(
                            step=global_step + 1,
                            loss=loss.item() * grad_accum_steps,
                            val_loss=val_loss,
                        )

                    global_step += 1

                epoch_loss += loss.item() * grad_accum_steps
                progress_bar.set_postfix(
                    {"loss": f"{loss.item() * grad_accum_steps:.4f}"}
                )

            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

            # Save Checkpoint at the end of each epoch
            # Use _orig_mod to save the raw state_dict (without torch.compile prefixes)
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved at {checkpoint_path}")

            # Sync Checkpoint to S3 (before metrics, so checkpoint is never lost)
            try:
                self.s3_client.upload_file(
                    str(checkpoint_path), self.bucket_name, "checkpoint.pt"
                )
                logger.info(
                    f"Checkpoint synced to S3: s3://{self.bucket_name}/checkpoint.pt"
                )
            except ClientError as e:
                logger.error(f"Failed to sync checkpoint to S3: {e}")

            # Save metrics to JSON (non-critical, after S3 sync)
            try:
                metrics_path = checkpoint_dir / "metrics.json"
                analyzer.save_metrics(metrics_path)
                analyzer.print_summary(epoch=epoch + 1, total_epochs=epochs)
                logger.info(f"Metrics saved at {metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to save metrics (non-critical): {e}")

        # 5. Save Tokenizer
        output_dir = checkpoint_dir / "tfm_slm_v1"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Tokenizer saved in: {output_dir}")
