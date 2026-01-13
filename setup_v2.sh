#!/bin/bash
set -e

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

ENV_NAME="deepseek-ocr-v2"
PYTHON_VERSION="3.12.9"

log_info "Checking for Conda..."
if ! command -v conda &> /dev/null; then
    log_error "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

log_info "Setting up Conda environment '$ENV_NAME'..."
if conda info --envs | grep -q "^$ENV_NAME "; then
    log_info "Environment '$ENV_NAME' already exists."
else
    log_info "Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" pip -y
fi

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

log_info "Installing dependencies..."
pip install --upgrade pip

if [[ "$(uname)" == "Linux" ]]; then
    log_info "Detected Linux. Following strict DeepSeek-OCR setup instructions (CUDA 11.8)..."
    
    # 1. Install PyTorch with cu118 (Matching setup.sh)
    log_info "Installing PyTorch (cu118)..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   
    # 2. Install vLLM from specific wheel (Matching setup.sh)
    VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl"
    log_info "Installing vLLM from wheel: $VLLM_WHEEL_URL"
    pip install "$VLLM_WHEEL_URL"

    # 3. Install flash-attn
    log_info "Installing flash-attn..."
    pip install flash-attn==2.7.3 --no-build-isolation

else
    log_info "Detected macOS. Using macOS-compatible versions (MPS support)..."

    # 1. Install PyTorch (Matching setup.sh)
    log_info "Installing PyTorch..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

    # 2. Install vLLM (Matching setup.sh)
    log_info "Installing vLLM..."
    pip install "vllm>=0.8.5"

    # 3. Install flash-attn (Best effort)
    log_info "Attempting to install flash-attn (may fail or require compilation on macOS)..."
    if ! pip install flash-attn==2.7.3 --no-build-isolation; then
         log_warning "flash-attn installation failed. Proceeding without it. This is expected on many macOS setups."
    fi
fi

# Install Dependencies and Fixes (Matching setup.sh logic)
log_info "Installing core dependencies (migrated from requirements.txt)..."
# We rely on pip install -e . to pull standard deps, but we force the fix first
# fix: requirements.txt pins tokenizers==0.20.3 which breaks vllm 0.6.x+
log_info "Fixing dependency conflict: Upgrading tokenizers and transformers for vLLM compatibility..."
pip install --upgrade "tokenizers>=0.21.0" transformers

# Install the library
log_info "Installing 'ldf-deepseek-ocr' library..."
pip install -e .

log_info "Setup complete!"
log_info "To use the environment, run: source init_v2.sh"
