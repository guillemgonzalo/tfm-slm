# Stage 1: Build environment — python:3.13 matches runtime exactly
# No CUDA needed: flash-attn installed from pre-built wheel
FROM python:3.13 AS builder

RUN pip install --upgrade pip uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./
COPY app/ ./app/

# Install project + deps (creates scripts in .venv/bin)
RUN uv sync --frozen --no-dev

# Install flash-attn pre-built wheel into the venv (only on x86_64 architectures)
RUN if [ "$(uname -m)" = "x86_64" ]; then \
      uv pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"; \
    else \
      echo "Skipping flash-attention pre-built wheel on non-x86_64 architecture"; \
    fi


# Stage 2: Runtime — same python:3.13-slim, .venv is fully compatible
FROM python:3.13-slim

# System deps for torch.compile (Triton)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY app/ /app/app/
COPY pyproject.toml uv.lock entrypoint.sh /app/

RUN chmod +x /app/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.13/site-packages/torch/lib:/app/.venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib"

ENV MODE=train

ENTRYPOINT ["/app/entrypoint.sh"]
