#!/bin/bash
set -e

# Configuration
REPO_URL="https://github.com/deepseek-ai/DeepSeek-OCR.git"
SUBMODULE_PATH="external/DeepSeek-OCR"
VENV_DIR=".venv"

# Function to print colored status messages
log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_warning() {
    echo -e "\033[0;33m[WARNING]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# 1. Setup Git Submodule
log_info "Setting up git submodule..."
if [ ! -d "$SUBMODULE_PATH" ]; then
    mkdir -p external
    git submodule add "$REPO_URL" "$SUBMODULE_PATH"
else
    log_info "Submodule directory exists. Updating..."
fi
git submodule update --init --recursive

# 2. Setup Conda Environment
ENV_NAME="deepseek-ocr"
PYTHON_VERSION="3.12.9"

log_info "Checking for Conda..."
if ! command -v conda &> /dev/null; then
    log_error "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

log_info "Setting up Conda environment '$ENV_NAME'..."
# Check if environment exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    log_info "Environment '$ENV_NAME' already exists."
else
    log_info "Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate Conda environment in this script
# This is necessary because 'conda activate' inside a script requires shell hook initialization
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 3. Install Dependencies
log_info "Installing dependencies in Conda environment: $ENV_NAME"
pip install --upgrade pip

if [[ "$(uname)" == "Linux" ]]; then
    log_info "Detected Linux. Following strict DeepSeek-OCR setup instructions (CUDA 11.8)..."
    
    # 1. Install PyTorch with cu118
    log_info "Installing PyTorch (cu118)..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

    # 2. Install vLLM from specific wheel
    # Using the URL directly to avoid manual download step
    VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl"
    log_info "Installing vLLM from wheel: $VLLM_WHEEL_URL"
    pip install "$VLLM_WHEEL_URL"

    # 3. Install Requirements
    log_info "Installing requirements.txt..."
    if [ -f "$SUBMODULE_PATH/requirements.txt" ]; then
        # Use --no-deps for tokenizers/transformers in requirements to avoid downgrading what vLLM needs?
        # Or just let it install and then fix it.
        pip install -r "$SUBMODULE_PATH/requirements.txt"
        
        # fix: requirements.txt pins tokenizers==0.20.3 which breaks vllm 0.6.x+
        log_info "Fixing dependency conflict: Upgrading tokenizers and transformers for vLLM compatibility..."
        pip install --upgrade "tokenizers>=0.21.0" transformers
    fi

    # 4. Install flash-attn
    log_info "Installing flash-attn..."
    pip install flash-attn==2.7.3 --no-build-isolation

else
    log_info "Detected macOS. Using macOS-compatible versions (MPS support)..."

    # 1. Install PyTorch (Standard PyPI has macOS binaries)
    log_info "Installing PyTorch..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

    # 2. Install vLLM (PyPI version)
    log_info "Installing vLLM..."
    pip install "vllm>=0.8.5"

    # 3. Install Requirements
    log_info "Installing requirements.txt..."
    if [ -f "$SUBMODULE_PATH/requirements.txt" ]; then
        pip install -r "$SUBMODULE_PATH/requirements.txt"
        
        # fix: same fix for mac
        log_info "Fixing dependency conflict: Upgrading tokenizers and transformers..."
        pip install --upgrade "tokenizers>=0.21.0" transformers
    fi

    # 4. Install flash-attn (Optional/Best-effort on Mac)
    log_info "Attempting to install flash-attn (may fail or require compilation on macOS)..."
    if ! pip install flash-attn==2.7.3 --no-build-isolation; then
        log_warning "flash-attn installation failed. Proceeding without it. This is expected on many macOS setups."
    fi
fi

log_info "Setup complete! Activate the environment with: source .venv/bin/activate"
