#!/bin/bash

# Configuration
ENV_NAME="deepseek-ocr-v2"

# Activate Conda Environment
# We assume conda is available in shell.
# If invalid, we try to source conda.sh if standard path
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    # Added based on user path observation
    source "$HOME/miniforge3/etc/profile.d/conda.sh" 
fi

# Detect if we successfully can activate
if command -v conda &> /dev/null; then
    conda activate "$ENV_NAME"
    echo "Activated conda environment: $ENV_NAME"
else
    echo "Error: Conda not found or not initialized in this shell."
    echo "Please ensure 'conda' is in your PATH or run 'conda init' first."
fi

# Set Environment Variables for DeepSeek-OCR Compatibility
export VLLM_USE_V1=0
echo "Set VLLM_USE_V1=0 (DeepSeek-OCR Compatibility)"

echo "Environment initialized. You can now run:"
echo "  python ocr_workflow_v2.py --input_dir ... --output_dir ..."
