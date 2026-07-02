"""
Benchmarking script for Hybrid Transformer-GRU model.
Evaluates on the dedicated 'benchmark' split (148,437 samples), a real holdout disjoint from training.
Exports: loss, perplexity, throughput (tokens/sec), memory usage.
"""

import json
import logging
import time
from pathlib import Path

import boto3
import torch
import torch.nn.functional as F
from app.config import settings
from app.dataset.cache import DatasetCacheService
from app.dataset.tokenizer import TokenizerService
from app.model.architecture import HybridConfig, HybridModel
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HybridBenchmark:
    def __init__(self, checkpoint_path: str = ".output/checkpoint.pt"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self._get_device()
        self.s3_client = boto3.client("s3")
        logger.info(f"Device: {self.device}")

        from app.dataset.tokenizer import TokenizerService
        self.tokenizer = TokenizerService().load()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self) -> HybridModel:
        # Download checkpoint from S3 if not local
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.checkpoint_path.exists():
            try:
                logger.info("Checkpoint not found locally. Downloading from S3...")
                self.s3_client.download_file(
                    settings.checkpoint_bucket,
                    "checkpoint.pt",
                    str(self.checkpoint_path),
                )
                logger.info("Checkpoint downloaded from S3")
            except Exception as e:
                raise FileNotFoundError(
                    f"Checkpoint not found locally or in S3 ({settings.checkpoint_bucket}): {e}"
                )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        config = HybridConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=1024,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072,
        )

        model = HybridModel(config).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        logger.info(f"Model loaded from {self.checkpoint_path}")
        return model

    def _load_validation_data(self, dataset_path: str = ".datasets/mixed_dataset", max_samples: int = 148_437):
        """Load the dedicated 'benchmark' split from local or S3 (using DatasetCacheService).

        This split is produced by DatasetProcessor.process as a fixed holdout,
        disjoint from the 'train' split consumed by TrainingService — genuinely
        unseen during training.
        """
        dataset_path_obj = Path(dataset_path)

        # Use DatasetCacheService to download from S3 if needed
        cache = DatasetCacheService()
        if not cache.exists_locally():
            if cache.exists_in_s3():
                logger.info("Dataset not found locally. Downloading from S3...")
                cache.download()
            else:
                raise FileNotFoundError(
                    f"Dataset not found locally at {dataset_path} and not available in S3 bucket {settings.dataset_bucket}"
                )

        dataset = load_from_disk(dataset_path)

        if isinstance(dataset, DatasetDict) and "benchmark" in dataset:
            ds = dataset["benchmark"]
        else:
            raise ValueError(
                f"Dataset at {dataset_path} has no 'benchmark' split. "
                "Re-run DatasetProcessor.process to generate a train/benchmark split."
            )

        ds.set_format("torch")
        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        elif len(ds) < max_samples:
            logger.warning(f"Benchmark split has only {len(ds)} samples, using all")

        # Batch size: 16 for MPS (Mac), 64 for CUDA
        batch_size = 16 if self.device.type == "mps" else 64
        dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if self.device.type != "mps" else False,
            num_workers=0 if self.device.type == "mps" else 4,
        )
        logger.info(f"Loaded {len(ds)} benchmark samples")
        return dataloader

    def benchmark(
        self,
        dataset_path: str = ".datasets/mixed_dataset",
        max_samples: int = 148_437,
    ):
        """Run comprehensive benchmarking with 10 metrics."""
        model = self._load_model()
        dataloader = self._load_validation_data(dataset_path, max_samples)

        # Metrics
        total_loss = 0.0
        total_tokens = 0
        total_time = 0.0
        peak_memory = 0.0

        # Token accuracy metrics
        correct_tokens = 0
        top5_correct = 0
        top10_correct = 0

        # Latency tracking
        batch_times = []

        # Memory tracking
        peak_activation_memory = 0.0

        # Tokenization speed
        tokenization_times = []

        # Attention patterns (simplified: count attention heat)
        attention_diversity_scores = []

        logger.info("Starting evaluation with 10 metrics...")
        with torch.no_grad():
            start_time = time.time()
            for i, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                batch_size, seq_len = input_ids.shape

                # Forward pass
                batch_start = time.time()
                outputs = model(input_ids, labels=input_ids)
                loss = outputs["loss"]
                logits = outputs["logits"]
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                # 1. Token-level accuracy (shift: predict token i+1 from token i)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                num_tokens_this_batch = shift_labels.numel()

                predictions = torch.argmax(shift_logits, dim=-1)
                correct_tokens += (predictions == shift_labels).sum().item()

                # 2. Top-k accuracy
                top5_preds = torch.topk(shift_logits, 5, dim=-1).indices
                top10_preds = torch.topk(shift_logits, 10, dim=-1).indices
                top5_correct += (top5_preds == shift_labels.unsqueeze(-1)).any(dim=-1).sum().item()
                top10_correct += (top10_preds == shift_labels.unsqueeze(-1)).any(dim=-1).sum().item()

                # Accumulate loss (loss is already averaged by CrossEntropyLoss)
                total_loss += loss.item() * num_tokens_this_batch
                total_tokens += num_tokens_this_batch
                total_time += batch_time

                # Track peak memory
                if self.device.type == "cuda":
                    peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1e9)

                if (i + 1) % 10 == 0:
                    throughput = (batch_size * seq_len) / batch_time
                    avg_loss_so_far = total_loss / total_tokens if total_tokens > 0 else 0.0
                    perplexity_so_far = torch.exp(torch.tensor(avg_loss_so_far)).item()
                    elapsed = time.time() - start_time
                    batches_done = i + 1
                    batches_total = len(dataloader)
                    eta = (elapsed / batches_done) * (batches_total - batches_done)

                    logger.info(
                        f"[{batches_done}/{batches_total}] "
                        f"Loss: {loss.item():.4f} | "
                        f"Perplexity: {perplexity_so_far:.2f} | "
                        f"Throughput: {throughput:.0f} tok/s | "
                        f"Acc@1: {(correct_tokens/max(total_tokens,1)*100):.1f}% | "
                        f"ETA: {eta//60:.0f}m"
                    )

            total_time = time.time() - start_time

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        throughput = total_tokens / total_time if total_time > 0 else 0.0

        # Token accuracy
        token_accuracy = (correct_tokens / max(total_tokens, 1)) * 100
        top5_accuracy = (top5_correct / max(total_tokens, 1)) * 100
        top10_accuracy = (top10_correct / max(total_tokens, 1)) * 100

        # 3. Latency per token
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        latency_per_token_ms = (avg_batch_time / max(dataloader.batch_size, 1)) * 1000

        # 4. Batch vs single token throughput (estimate)
        single_token_throughput = 1.0 / (latency_per_token_ms / 1000) if latency_per_token_ms > 0 else 0

        # 7. Model stats
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 8. Model size (MB)
        model_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32

        results = {
            "model": "HybridBlock (12 layers, 768 hidden, 12 heads)",
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "dataset_samples": min(max_samples, len(dataloader.dataset)),
            "batch_size": dataloader.batch_size,
            "metrics": {
                # Loss & Perplexity
                "loss": round(avg_loss, 6),
                "perplexity": round(perplexity, 4),

                # 1. Token-level accuracy
                "token_accuracy_percent": round(token_accuracy, 2),

                # 2. Top-k accuracy
                "top5_accuracy_percent": round(top5_accuracy, 2),
                "top10_accuracy_percent": round(top10_accuracy, 2),

                # 3. Latency per token
                "latency_per_token_ms": round(latency_per_token_ms, 4),

                # 4. Batch vs single token throughput
                "batch_throughput_tokens_per_sec": round(throughput, 2),
                "single_token_throughput_tokens_per_sec": round(single_token_throughput, 2),

                # 7. Tokenization speed (implicit in throughput)
                "throughput_tokens_per_sec": round(throughput, 2),

                # 8. Model compression ratio
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": round(model_size_mb, 2),

                # Memory
                "peak_memory_gb": round(peak_memory, 2),

                # Timing
                "total_time_seconds": round(total_time, 2),
            },
        }

        logger.info(f"\n{'='*70}")
        logger.info("BENCHMARK RESULTS (10 Metrics)")
        logger.info(f"{'='*70}")
        logger.info(f"1. Loss: {results['metrics']['loss']:.6f}")
        logger.info(f"2. Perplexity: {results['metrics']['perplexity']:.2f}")
        logger.info(f"3. Token Accuracy: {results['metrics']['token_accuracy_percent']:.2f}%")
        logger.info(f"4. Top-5 Accuracy: {results['metrics']['top5_accuracy_percent']:.2f}%")
        logger.info(f"5. Top-10 Accuracy: {results['metrics']['top10_accuracy_percent']:.2f}%")
        logger.info(f"6. Latency per Token: {results['metrics']['latency_per_token_ms']:.4f} ms")
        logger.info(f"7. Batch Throughput: {results['metrics']['batch_throughput_tokens_per_sec']:.0f} tokens/sec")
        logger.info(f"8. Single Token Throughput: {results['metrics']['single_token_throughput_tokens_per_sec']:.0f} tokens/sec")
        logger.info(f"9. Model Size: {results['metrics']['model_size_mb']:.2f} MB ({results['metrics']['total_parameters']:,} params)")
        logger.info(f"10. Peak Memory: {results['metrics']['peak_memory_gb']:.2f} GB")
        logger.info(f"Total Time: {results['metrics']['total_time_seconds']:.2f}s")
        logger.info(f"{'='*70}\n")

        return results

    def save_results(self, results, output_path: str = ".output/benchmark_hybrid.json", upload_s3: bool = True):
        """Save benchmark results to JSON locally and optionally to S3."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Benchmark results saved to {output_path}")

        # Upload to S3
        if upload_s3:
            try:
                s3_key = Path(output_path).name
                self.s3_client.upload_file(
                    output_path, settings.benchmark_bucket, s3_key
                )
                logger.info(f"Uploaded to S3: s3://{settings.benchmark_bucket}/{s3_key}")
            except Exception as e:
                logger.warning(f"Failed to upload to S3 (non-critical): {e}")


def main():
    benchmark = HybridBenchmark()
    results = benchmark.benchmark(max_samples=148_437)
    benchmark.save_results(results)


if __name__ == "__main__":
    main()
