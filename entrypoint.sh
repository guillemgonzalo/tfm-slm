#!/bin/bash
set -e

MODE=${MODE:-train}

echo "Starting TFM-SLM in $MODE mode..."

if [ "$MODE" = "inference" ]; then
    echo "Running inference API server..."
    tfm-slm-api

elif [ "$MODE" = "benchmark" ]; then
    echo "Running benchmarking..."
    tfm-slm-benchmark
else
    echo "Running training..."
    tfm-slm
fi
