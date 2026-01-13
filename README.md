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
*   `--schema_file`: (Optional) Path to a JSON file containing the schema for structured data extraction.

### 3. Run the OCR Application

The application processes all PDF files in the input directory and saves the results to the output directory. It is crucial to set `VLLM_USE_V1=0` to ensure compatibility.

**Example 1: Organization Chart Extraction**

```bash
VLLM_USE_V1=0 python ocr_app.py \
    --input_dir input/orgchart \
    --output_dir output/orgchart \
    --prompt_file input/orgchart/prompt.txt \
    --schema_file input/orgchart/schema.json
```

**Example 2: Tourism Data Extraction**

```bash
VLLM_USE_V1=0 python ocr_app.py \
    --input_dir input/tourism \
    --output_dir output/tourism \
    --prompt_file input/tourism/prompt.txt \
    --schema_file input/tourism/schema.json
```

**Note:** The `--schema_file` argument is optional but recommended for structured metadata extraction. It appends the schema content to your prompt.

## Multi-Agent Workflow (Advanced)

For more complex use cases requiring structural formatting (JSON) and multi-page data consolidation, use the `ocr_workflow.py`. This script employs a Multi-Agent architecture:

1.  **Extractor**: Runs DeepSeek-OCR to get raw text.
2.  **Processor**: Uses your custom `prompt.txt` to structure the raw text into JSON.
3.  **Aggregator**: Stitches pages together (handling "CONTINUATION" logic).
4.  **Finalizer**: Saves the consolidated data.

### Usage
```bash
python ocr_workflow.py \
    --input_dir input/orgchart \
    --output_dir output/orgchart_workflow \
    --prompt_file input/orgchart/prompt.txt
```
**Note:** This workflow re-uses the loaded DeepSeek model for both OCR and post-processing, optimizing VRAM usage.

## Library Setup (V2)

We have refactored the project into a proper python library `ldf.deepseek.ocr`.

### 1. Setup

Run the V2 setup script to create a new environment (`deepseek-ocr-v2`) and install the library:

```bash
./setup_v2.sh
```

### 2. Initialization

To activate the environment and set necessary variables:

```bash
source init_v2.sh
```

### 3. Running Workflow V2

Use the new `ocr_workflow_v2.py` script which utilizes the library structure:

```bash
python ocr_workflow_v2.py \
    --input_dir input/orgchart \
    --output_dir output/orgchart_workflow_v2 \
    --prompt_file input/orgchart/prompt.txt
```

## Credits

This library is heavily based on and contains code from the [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) repository.
We gratefully acknowledge the DeepSeek-AI team for their open-source work.

Original Repository: https://github.com/deepseek-ai/DeepSeek-OCR
License: Please refer to the original repository for license details.
