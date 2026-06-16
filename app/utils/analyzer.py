import json
import logging
from pathlib import Path
from typing import Dict, List, Optional  # noqa: UP035

logger = logging.getLogger(__name__)


class HybridArchitectureAnalyzer:
    """
    Analyzes component contributions (GRU, Attention, MLP) during training.
    Tracks metrics, logs, and exports analysis results.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.metrics_history = []

    def analyze_component_contributions(self, input_ids):
        """
        Analyze norm of gradients for GRU, Attention, and MLP components.
        Returns dict with component norms and GRU/Attn ratio.
        """
        gru_norm = 0.0
        attn_norm = 0.0
        mlp_norm = 0.0

        for block in self.model.blocks:
            for name, param in block.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.norm().item() ** 2
                    if 'gru' in name:
                        gru_norm += param_norm
                    elif 'attn' in name:
                        attn_norm += param_norm
                    elif 'mlp' in name:
                        mlp_norm += param_norm

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

    def analyze_hidden_state_flow(self):
        """
        Analyze how GRU hidden states evolve across layers.
        Returns statistics about hidden state norms.
        """
        if not hasattr(self.model, 'gru_hidden_states'):
            return None

        hidden_states = self.model.gru_hidden_states
        if hidden_states is None or len(hidden_states) == 0:
            return None

        norms = []
        for h_state in hidden_states:
            if h_state is not None:
                norms.append(h_state.norm().item())

        if not norms:
            return None

        return {
            "min_norm": min(norms),
            "max_norm": max(norms),
            "mean_norm": sum(norms) / len(norms),
            "num_layers": len(norms),
        }

    def log_metrics(
        self,
        step: int,
        loss: float,
        val_loss: Optional[float] = None,
        grad_norm_before: Optional[float] = None,
        grad_norm_after: Optional[float] = None,
        component_metrics: Optional[Dict] = None,
    ):
        """
        Log metrics for a training step.
        """
        metric_entry = {
            "step": step,
            "loss": loss,
            "val_loss": val_loss,
            "grad_norm_before": grad_norm_before,
            "grad_norm_after": grad_norm_after,
        }

        if component_metrics:
            metric_entry.update(component_metrics)

        self.metrics_history.append(metric_entry)

    def save_metrics(self, output_path: Path):
        """
        Save metrics history to JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import torch

        def _serialize(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(output_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=_serialize)

        logger.info(f"Metrics saved to {output_path}")

    def get_summary_stats(self) -> Dict:
        """
        Compute summary statistics from metrics history.
        """
        if not self.metrics_history:
            return {}

        losses = [m["loss"] for m in self.metrics_history if "loss" in m]
        val_losses = [m["val_loss"] for m in self.metrics_history if m.get("val_loss")]
        gru_ratios = [m["gru_to_attn_ratio"] for m in self.metrics_history if "gru_to_attn_ratio" in m]

        all_steps = [m["step"] for m in self.metrics_history if "step" in m]
        stats = {
            "num_steps": max(all_steps) if all_steps else 0,
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "max_loss": max(losses) if losses else None,
            "mean_loss": sum(losses) / len(losses) if losses else None,
        }

        if val_losses:
            stats["final_val_loss"] = val_losses[-1]
            stats["min_val_loss"] = min(val_losses)
            stats["mean_val_loss"] = sum(val_losses) / len(val_losses)

        if gru_ratios:
            stats["mean_gru_ratio"] = sum(gru_ratios) / len(gru_ratios)
            stats["min_gru_ratio"] = min(gru_ratios)
            stats["max_gru_ratio"] = max(gru_ratios)

        return stats

    def print_summary(self, epoch: int | None = None, total_epochs: int | None = None):
        """
        Print human-readable summary of analysis.
        """
        stats = self.get_summary_stats()

        logger.info("=" * 60)
        logger.info("HYBRID ARCHITECTURE ANALYSIS SUMMARY")
        logger.info("=" * 60)

        if epoch is not None and total_epochs is not None:
            logger.info(f"Epoch: {epoch}/{total_epochs}")
        logger.info(f"Steps trained: {stats.get('num_steps', 'N/A')}")
        logger.info(f"Final loss: {stats.get('final_loss', 'N/A'):.4f}")
        logger.info(f"Min loss: {stats.get('min_loss', 'N/A'):.4f}")
        logger.info(f"Mean loss: {stats.get('mean_loss', 'N/A'):.4f}")

        if "final_val_loss" in stats:
            logger.info(f"Final validation loss: {stats['final_val_loss']:.4f}")
            logger.info(f"Min validation loss: {stats['min_val_loss']:.4f}")

        if "mean_gru_ratio" in stats:
            logger.info(f"GRU/Attn ratio (mean): {stats['mean_gru_ratio']:.3f}")
            logger.info(f"GRU/Attn ratio (range): {stats['min_gru_ratio']:.3f} - {stats['max_gru_ratio']:.3f}")
            if 0.5 <= stats["mean_gru_ratio"] <= 1.5:
                logger.info("✅ GRU/Attn ratio: WELL BALANCED")
            elif stats["mean_gru_ratio"] < 0.3:
                logger.info("⚠️ GRU/Attn ratio: GRU may be too weak")
            elif stats["mean_gru_ratio"] > 1.5:
                logger.info("⚠️ GRU/Attn ratio: GRU may be dominating")

        logger.info("=" * 60)

    def compare_architectures(self, baseline_metrics: Optional[Dict] = None) -> Dict:
        """
        Compare current training with baseline (if provided).
        """
        current_stats = self.get_summary_stats()

        if baseline_metrics is None:
            return {"current": current_stats}

        comparison = {
            "current": current_stats,
            "baseline": baseline_metrics,
            "improvements": {},
        }

        if "mean_loss" in current_stats and "mean_loss" in baseline_metrics:
            loss_diff = baseline_metrics["mean_loss"] - current_stats["mean_loss"]
            loss_pct = (loss_diff / baseline_metrics["mean_loss"]) * 100 if baseline_metrics["mean_loss"] != 0 else 0
            comparison["improvements"]["loss"] = {
                "absolute": loss_diff,
                "percentage": loss_pct,
            }

        return comparison
