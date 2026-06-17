import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


class HybridConfig(PretrainedConfig):
    model_type = "hybrid_transformer_gru"

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024,
        dropout=0.1,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings


class FlashAttention(nn.Module):
    """
    Flash Attention wrapper for 2-3x speedup on modern NVIDIA GPUs.
    Reduces memory from O(n²) to O(n) and is much faster due to I/O-aware optimization.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = hidden_size // num_heads

        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, past_kv=None, use_cache=False):
        batch_size, seq_len_q, _ = q.shape

        q = self.Wq(q).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.Wk(k).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        v = self.Wv(v).view(batch_size, seq_len_q, self.num_heads, self.head_dim)

        # KV-cache: prepend previously computed keys/values (layout [b, seq, heads, head_dim])
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present = (k, v) if use_cache else None
        seq_len_kv = k.shape[1]

        if _HAS_FLASH_ATTN:
            # flash_attn causal=True uses bottom-right alignment, correct for q_len <= kv_len
            out = flash_attn_func(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            )
            out = out.contiguous().view(batch_size, seq_len_q, self.hidden_size)
        else:
            # Fallback: scaled_dot_product_attention expects [batch, heads, seq, head_dim]
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)

            # Causal mask aligned to the end: query i (absolute pos offset+i) attends key j<=offset+i.
            # Bool mask semantics: True = allowed to attend.
            offset = seq_len_kv - seq_len_q
            q_idx = torch.arange(seq_len_q, device=q.device).unsqueeze(1) + offset
            k_idx = torch.arange(seq_len_kv, device=q.device).unsqueeze(0)
            allowed = k_idx <= q_idx
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
            )
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_size)

        out = self.out_proj(out)
        return out, present


class HybridBlock(nn.Module):
    """
    A single block that combines GRU (pre-attention), Self-Attention, and FFN.
    Order: GRU → Attention → MLP. Returns (output, gru_hidden) for persistent states.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.attn = FlashAttention(config.hidden_size, config.num_heads, config.dropout)

        self.ln_3 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, gru_hidden=None, past_kv=None, use_cache=False):
        # 1. GRU (Sequential/Local Refinement - Pre-Attention)
        residual = x
        x = self.ln_1(x)
        gru_output, gru_hidden = self.gru(x, gru_hidden)
        x = residual + gru_output

        # 2. Multi-Head Attention (Global Context) - Flash Attention + KV-cache
        residual = x
        x = self.ln_2(x)
        attn_output, present = self.attn(x, x, x, past_kv=past_kv, use_cache=use_cache)
        x = residual + attn_output

        # 3. Feed Forward (Feature Refinement)
        residual = x
        x = self.ln_3(x)
        x = residual + self.mlp(x)

        return x, gru_hidden, present


class HybridModel(PreTrainedModel):
    config_class = HybridConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [HybridBlock(config) for _ in range(config.num_layers)]
        )

        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

        # Pre-compute and register causal mask as a buffer
        # This prevents recreating it and moving it from CPU to GPU every forward pass
        mask = torch.triu(
            torch.ones(config.max_position_embeddings, config.max_position_embeddings),
            diagonal=1,
        ).bool()
        self.register_buffer("causal_mask", mask)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

    def forward(
        self,
        input_ids,
        labels=None,
        gru_hidden_states=None,
        past_key_values=None,
        use_cache=False,
    ):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Absolute position offset from cached length (incremental decoding)
        past_length = (
            past_key_values[0][0].shape[1]
            if past_key_values is not None and past_key_values[0] is not None
            else 0
        )

        # Embeddings
        positions = torch.arange(
            past_length, past_length + seq_length, device=device
        ).unsqueeze(0)
        x = self.embeddings(input_ids) + self.pos_embeddings(positions)
        x = self.dropout(x)

        # Initialize persistent GRU + KV states if not provided
        if gru_hidden_states is None:
            gru_hidden_states = [None] * len(self.blocks)
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        new_gru_hidden_states = []
        new_key_values = [] if use_cache else None

        # Hybrid blocks with persistent hidden states + KV-cache
        for i, block in enumerate(self.blocks):
            x, new_hidden, present = block(
                x,
                gru_hidden=gru_hidden_states[i],
                past_kv=past_key_values[i],
                use_cache=use_cache,
            )
            new_gru_hidden_states.append(new_hidden)
            if use_cache:
                new_key_values.append(present)

        self.gru_hidden_states = new_gru_hidden_states

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        out = {"logits": logits}
        if loss is not None:
            out["loss"] = loss
        if use_cache:
            out["past_key_values"] = new_key_values
            out["gru_hidden_states"] = new_gru_hidden_states
        return out

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Greedy/Top-k generation helper for inference.
        """
        self.eval()

        # Prefill: process the full prompt once, then decode token-by-token from cache.
        input_cond = input_ids[:, -self.config.max_position_embeddings :]
        past_key_values = None
        gru_hidden_states = None
        next_input = input_cond

        for _ in range(max_new_tokens):
            outputs = self(
                next_input,
                past_key_values=past_key_values,
                gru_hidden_states=gru_hidden_states,
                use_cache=True,
            )
            past_key_values = outputs["past_key_values"]
            gru_hidden_states = outputs["gru_hidden_states"]

            logits = outputs["logits"][:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)
            # Only the new token flows through the next step; cache holds the rest.
            next_input = next_token

        return input_ids
