# HuggingFace DeepSeek OCR Workspace

This repository is set up to work with [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR).

## Setup

**Prerequisites:** You must have [Conda](https://docs.conda.io/en/latest/) (Anaconda or Miniconda) installed.

To set up the environment and dependencies, simply run the setup script:

```bash
./setup.sh
```

This script will:
1.  Initialize the `DeepSeek-OCR` submodule in `external/DeepSeek-OCR`.
2.  Create a Conda environment named `deepseek-ocr` with Python 3.12.9.
3.  Install necessary dependencies including `torch`, `vllm`, and `DeepSeek-OCR` requirements.

## Usage

After setup, activate the environment:

```bash
conda activate deepseek-ocr
```

You can then access the `DeepSeek-OCR` code in `external/DeepSeek-OCR`.

## Running the OCR Application

This repository includes a custom application `ocr_app.py` for batch processing PDF files.

> [!NOTE]
> DeepSeek-OCR requires custom logits processors which are not supported in vLLM V1. The application defaults to `VLLM_USE_V1=0` to ensure compatibility.

### Basic Usage

```bash
python ocr_app.py --input_dir /path/to/pdfs --output_dir /path/to/output
```

### Using a Custom Prompt

You can provide a specific prompt via a text file:

```bash
python ocr_app.py --input_dir /input/path --output_dir /output/path --prompt_file /path/to/prompt.txt
```

**Arguments:**
*   `--input_dir`: Directory containing `.pdf` files.
*   `--output_dir`: Directory where extracted JSON and images will be saved.
*   `--prompt_file`: (Optional) Path to a text file containing the custom prompt.

## Examples

### Orgchart Example

To parse the organizational charts and extract minister data:

```bash
VLLM_USE_V1=0 python ocr_app.py --input_dir input/orgchart --output_dir output/orgchart --prompt_file input/orgchart/prompt.txt
```

### Tourism Example

To extract tables and metadata from the tourism report:

```bash
VLLM_USE_V1=0 python ocr_app.py --input_dir input/tourism --output_dir output/tourism --prompt_file input/tourism/prompt.txt
```
