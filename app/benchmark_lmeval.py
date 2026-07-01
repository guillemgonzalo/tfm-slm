"""
Zero-shot multiple-choice benchmarks for Hybrid Transformer-GRU model:
HellaSwag, ARC (Easy + Challenge), PIQA.

Method: log-likelihood scoring (standard lm-eval-harness approach).
For each example, score each answer choice by summing the model's
log-probabilities over the choice's tokens given the context, normalized
by token count. The choice with the highest normalized log-likelihood
is the prediction; accuracy is computed against the gold label.
"""

import json
import logging
from pathlib import Path

import boto3
import torch
import torch.nn.functional as F
from app.config import settings
from app.dataset.tokenizer import TokenizerService
from app.model.architecture import HybridConfig, HybridModel
from datasets import load_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_CONTEXT_TOKENS = 1024


class LMEvalBenchmark:
    def __init__(self, checkpoint_path: str = ".output/checkpoint.pt"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self._get_device()
        self.s3_client = boto3.client("s3")
        logger.info(f"Device: {self.device}")
        self.tokenizer = TokenizerService().load()
        self.model = self._load_model()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self) -> HybridModel:
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
            max_position_embeddings=MAX_CONTEXT_TOKENS,
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

    @torch.no_grad()
    def _choice_logprob(self, context: str, choice: str) -> float:
        """Sum of log-probs of `choice` tokens conditioned on `context`, averaged per token."""
        context_ids = self.tokenizer.encode(context)
        choice_ids = self.tokenizer.encode(choice)
        if len(choice_ids) == 0:
            return float("-inf")

        input_ids = context_ids + choice_ids
        input_ids = input_ids[-MAX_CONTEXT_TOKENS:]
        num_choice_tokens = min(len(choice_ids), len(input_ids))

        input_tensor = torch.tensor([input_ids], device=self.device)
        logits = self.model(input_tensor)["logits"][0]

        # logits[t] predicts token t+1
        pred_logits = logits[-num_choice_tokens - 1 : -1]
        target_ids = torch.tensor(input_ids[-num_choice_tokens:], device=self.device)

        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        return token_log_probs.mean().item()

    def _score_example(self, context: str, choices: list[str]) -> int:
        scores = [self._choice_logprob(context, choice) for choice in choices]
        return int(torch.tensor(scores).argmax().item())

    def run_hellaswag(self, limit: int | None = None) -> dict:
        ds = load_dataset("hellaswag", split="validation", trust_remote_code=settings.trust_remote_code)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        correct = 0
        for i, ex in enumerate(ds):
            context = ex["ctx"]
            choices = ex["endings"]
            label = int(ex["label"])
            pred = self._score_example(context, choices)
            correct += int(pred == label)
            if (i + 1) % 50 == 0:
                logger.info(f"HellaSwag [{i + 1}/{len(ds)}] running acc: {correct / (i + 1) * 100:.2f}%")

        return {"task": "hellaswag", "num_examples": len(ds), "accuracy_percent": round(correct / len(ds) * 100, 2)}

    def run_arc(self, subset: str = "ARC-Easy", limit: int | None = None) -> dict:
        ds = load_dataset("ai2_arc", subset, split="test", trust_remote_code=settings.trust_remote_code)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        correct = 0
        total = 0
        for i, ex in enumerate(ds):
            choices = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            answer_key = ex["answerKey"]
            if answer_key not in labels or len(choices) == 0:
                continue
            label = labels.index(answer_key)
            context = f"Question: {ex['question']}\nAnswer:"
            pred = self._score_example(context, choices)
            correct += int(pred == label)
            total += 1
            if (i + 1) % 50 == 0:
                logger.info(f"{subset} [{i + 1}/{len(ds)}] running acc: {correct / total * 100:.2f}%")

        return {"task": subset, "num_examples": total, "accuracy_percent": round(correct / total * 100, 2)}

    def run_piqa(self, limit: int | None = None) -> dict:
        ds = load_dataset("piqa", split="validation", trust_remote_code=settings.trust_remote_code)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        correct = 0
        for i, ex in enumerate(ds):
            context = f"Question: {ex['goal']}\nAnswer:"
            choices = [ex["sol1"], ex["sol2"]]
            label = int(ex["label"])
            pred = self._score_example(context, choices)
            correct += int(pred == label)
            if (i + 1) % 50 == 0:
                logger.info(f"PIQA [{i + 1}/{len(ds)}] running acc: {correct / (i + 1) * 100:.2f}%")

        return {"task": "piqa", "num_examples": len(ds), "accuracy_percent": round(correct / len(ds) * 100, 2)}

    def run_all(self, limit: int | None = None) -> dict:
        results = {
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "tasks": {},
        }
        for name, fn in [
            ("hellaswag", lambda: self.run_hellaswag(limit)),
            ("arc_easy", lambda: self.run_arc("ARC-Easy", limit)),
            ("arc_challenge", lambda: self.run_arc("ARC-Challenge", limit)),
            ("piqa", lambda: self.run_piqa(limit)),
        ]:
            logger.info(f"Running {name}...")
            result = fn()
            results["tasks"][name] = result
            logger.info(f"{name}: {result['accuracy_percent']:.2f}% ({result['num_examples']} examples)")

        return results

    def save_results(self, results: dict, output_path: str = ".output/benchmark_lmeval.json", upload_s3: bool = True):
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

        if upload_s3:
            try:
                s3_key = Path(output_path).name
                self.s3_client.upload_file(output_path, settings.benchmark_bucket, s3_key)
                logger.info(f"Uploaded to S3: s3://{settings.benchmark_bucket}/{s3_key}")
            except Exception as e:
                logger.warning(f"Failed to upload to S3 (non-critical): {e}")


def main():
    benchmark = LMEvalBenchmark()
    results = benchmark.run_all()
    benchmark.save_results(results)


if __name__ == "__main__":
    main()
