#!/bin/bash
set -e

# Function to print colored status messages
log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

# 1. Create Base Directory
log_info "Creating 'code' directory..."
mkdir -p code
cd code

# 2. Clone/Update Repository
REPO_DIR="hugging-face-deepseek-ocr"
log_info "Checking repository status..."

if [ ! -d "$REPO_DIR" ]; then
    log_info "Cloning repository..."
    git clone https://github.com/vibhatha/hugging-face-deepseek-ocr.git
else
    log_info "Repository exists. Updating..."
    cd "$REPO_DIR"
    git pull
    cd ..
fi

# 3. Run Setup
cd "$REPO_DIR"
chmod +x setup.sh

log_info "Launching setup.sh..."
./setup.sh
conda init
