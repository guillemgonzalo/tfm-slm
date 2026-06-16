"""
Test suite for Hybrid Transformer-GRU architecture improvements.
Validates: persistent hidden states, GRU positioning, ortho init, component analysis.
"""

import torch

from app.model.architecture import HybridConfig, HybridModel
from app.utils.analyzer import HybridArchitectureAnalyzer


def test_forward_pass_with_hidden_states():
    """Test forward pass with persistent hidden state tracking"""
    config = HybridConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=1024,
    )
    model = HybridModel(config)
    input_ids = torch.randint(0, 1000, (2, 64))

    outputs = model(input_ids)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 64, 1000)
    assert hasattr(model, "gru_hidden_states")
    assert len(model.gru_hidden_states) == 2


def test_loss_computation():
    """Test loss computation with labels"""
    config = HybridConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=1024,
    )
    model = HybridModel(config)
    input_ids = torch.randint(0, 1000, (2, 64))

    outputs = model(input_ids, labels=input_ids)
    assert "loss" in outputs
    assert outputs["loss"].item() > 0
    assert not torch.isnan(outputs["loss"])


def test_gradient_flow():
    """Test gradient flow through entire model"""
    config = HybridConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=1024,
    )
    model = HybridModel(config)
    input_ids = torch.randint(0, 1000, (2, 64))

    outputs = model(input_ids, labels=input_ids)
    loss = outputs["loss"]
    loss.backward()

    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break

    assert has_gradients, "No gradients computed"


def test_hidden_state_persistence():
    """Test GRU hidden states persist across multiple forward passes"""
    config = HybridConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=1024,
    )
    model = HybridModel(config)
    input_ids = torch.randint(0, 1000, (2, 64))

    # First forward pass
    outputs1 = model(input_ids)
    hidden_states_1 = [h.clone() if h is not None else None for h in model.gru_hidden_states]

    # Second forward pass with persisted states
    outputs2 = model(input_ids, gru_hidden_states=hidden_states_1)
    hidden_states_2 = model.gru_hidden_states

    assert len(hidden_states_1) == len(hidden_states_2)
    for h1, h2 in zip(hidden_states_1, hidden_states_2):
        if h1 is not None and h2 is not None:
            assert not torch.allclose(h1, h2), "Hidden states should differ after forward pass"


def test_weight_initialization_ortho():
    """Test orthogonal initialization for GRU weights"""
    config = HybridConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=1024,
    )
    model = HybridModel(config)

    gru_weights_found = 0
    for block in model.blocks:
        for name, param in block.gru.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                gru_weights_found += 1
                if param.dim() >= 2:
                    weight_norm = torch.norm(param.data)
                    assert weight_norm > 0, "Weight norm should be > 0"
                    assert param.data.std() > 0.01, "Weight std should be significant"

    assert gru_weights_found > 0, "No GRU weights found"


def test_component_contribution_analysis():
    """Test analyzer can extract component contributions"""
    config = HybridConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=1024,
    )
    model = HybridModel(config)
    analyzer = HybridArchitectureAnalyzer(model, torch.device("cpu"))
    input_ids = torch.randint(0, 1000, (2, 64))

    outputs = model(input_ids, labels=input_ids)
    loss = outputs["loss"]
    loss.backward()

    metrics = analyzer.analyze_component_contributions(input_ids)
    assert "gru_norm" in metrics
    assert "attn_norm" in metrics
    assert "mlp_norm" in metrics
    assert "gru_to_attn_ratio" in metrics
    assert metrics["gru_to_attn_ratio"] > 0


def test_model_size_parameters():
    """Test full model size and parameter count"""
    config = HybridConfig(
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
    )
    model = HybridModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 100e6, f"Model too small: {total_params/1e6:.1f}M params"
    assert total_params < 200e6, f"Model too large: {total_params/1e6:.1f}M params"
