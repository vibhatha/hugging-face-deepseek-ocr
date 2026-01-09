import os
import sys

# 0. Setup Environment Variables
# DeepSeek-OCR requires disabling VLLM V1 for custom logits processors
# We must ensure VLLM_USE_V1='0' is set in the process environment from the start.
if os.environ.get('VLLM_USE_V1') != '0':
    print("Restarting script with VLLM_USE_V1=0 to disable V1 engine...")
    new_env = os.environ.copy()
    new_env['VLLM_USE_V1'] = '0'
    new_env['CUDA_VISIBLE_DEVICES'] = '0'
    os.execve(sys.executable, [sys.executable] + sys.argv, new_env)

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

import glob
import json
import argparse
import time
import io
import re
from pathlib import Path 

# 1. Setup Path to import DeepSeek-OCR modules
current_dir = os.getcwd()
deepseek_vllm_path = os.path.join(current_dir, "external/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm")
sys.path.append(deepseek_vllm_path)

# Verify imports work
try:
    import torch
    from PIL import Image, ImageDraw, ImageFont
    import fitz
    import img2pdf
    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry
    
    # DeepSeek Imports
    from deepseek_ocr import DeepseekOCRForCausalLM
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    from process.image_process import DeepseekOCRProcessor
    import config # Import config to get defaults if needed
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Please define 'external/DeepSeek-OCR' submodule and run 'pip install -r external/DeepSeek-OCR/requirements.txt'")
    sys.exit(1)

# Register Model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def setup_llm():
    print("Initializing vLLM Engine...")
    # Use defaults from config or hardcode sensible ones
    model_path = config.MODEL_PATH
    # Ensure model path is correct or user provided. Defaulting to what's in config ('deepseek-ai/DeepSeek-OCR')
    
    llm = LLM(
        model=model_path,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=config.MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True
    )
    
    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    
    return llm, sampling_params

def pdf_to_images(pdf_path, dpi=144):
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            Image.MAX_IMAGE_PIXELS = None
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        pdf_document.close()
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
    return images

# Reuse the re_match logic from run_dpsk_ocr_pdf.py
def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other

def extract_coordinates(ref_text):
    try:
        # ref_text format roughly: <|ref|>image<|/ref|><|det|>[[x1,y1,x2,y2], ...]<|/det|>
        # But based on code: label_type = ref_text[1], cor_list = eval(ref_text[2])
        # wait, regex groups. 0 is full match. 1 is inside ref tags?
        # pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        # Group 1: full string? No.
        # matches returns tuples of groups.
        # Group 1: <|ref|>...<|/det|> (full match of the pattern parens) ? No, outer parens.
        # Group 2: inside ref
        # Group 3: inside det
        
        # Let's trust the logic if we parse it ourselves or use their function if we imported it.
        # Since I didn't import the Utils functions directly (they are file-level defs), I'll reimplement.
        pass
    except:
        pass

def crop_and_save_images(image, matches_images, output_dir, page_idx, base_filename):
    image_width, image_height = image.size
    saved_images = []
    
    img_idx = 0
    for match in matches_images:
        # match is the full string: <|ref|>image<|/ref|><|det|>[[...]]<|/det|>
        # Extract coordinates
        try:
            # We assume the regex in re_match returns full match string in group 0 if handled well, 
            # but the re_match implementation in original code returns a list of tuples?
            # pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
            # re.findall returns list of tuples [(full_match, ref_content, det_content)]
            
            # Let's re-parse match string or use the group content if passed
            
            # Simple manual parse for safety
            det_content_match = re.search(r'<\|det\|>(.*?)<\|/det\|>', match)
            if det_content_match:
                cor_list = eval(det_content_match.group(1))
                
                for points in cor_list:
                    x1, y1, x2, y2 = points
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)
                    
                    cropped = image.crop((x1, y1, x2, y2))
                    fname = f"{base_filename}_p{page_idx}_{img_idx}.jpg"
                    save_path = os.path.join(output_dir, fname)
                    cropped.save(save_path)
                    saved_images.append(save_path)
                    img_idx += 1
        except Exception as e:
            print(f"Error processing image crop: {e}")
            continue
            
    return saved_images

def parse_and_save_artifacts(text, output_dir, page_idx, base_filename):
    """
    Parses Markdown text for code blocks (csv, json) and saves them to files.
    """
    artifacts = {
        "csvs": [],
        "metadata": []
    }
    
    # Create subdirectories
    csv_dir = os.path.join(output_dir, "csvs")
    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    # Regex for code blocks: ```type content ```
    # Using dotall to capture multi-line content
    # We iterate to find all blocks
    
    # 1. Find CSV blocks
    # Pattern: ```csv\n(content)\n```
    csv_matches = re.finditer(r'```csv\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    for idx, match in enumerate(csv_matches):
        content = match.group(1)
        if content.strip():
            fname = f"{base_filename}_p{page_idx}_t{idx}.csv"
            save_path = os.path.join(csv_dir, fname)
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                artifacts["csvs"].append(os.path.relpath(save_path, output_dir))
            except Exception as e:
                print(f"Error saving CSV artifact: {e}")

    # 2. Find JSON blocks (Metadata)
    # Pattern: ```json\n(content)\n```
    json_matches = re.finditer(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    for idx, match in enumerate(json_matches):
        content = match.group(1)
        if content.strip():
            fname = f"{base_filename}_p{page_idx}_m{idx}.json"
            save_path = os.path.join(meta_dir, fname)
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                artifacts["metadata"].append(os.path.relpath(save_path, output_dir))
            except Exception as e:
                print(f"Error saving JSON artifact: {e}")
                
    return artifacts

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR PDF Processor")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output JSON and images")
    parser.add_argument("--prompt_file", type=str, help="Path to a text file containing the custom prompt", default=None)
    parser.add_argument("--schema_file", type=str, help="Path to a JSON/YAML file containing the metadata schema", default=None)
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return

    # Setup directories
    images_out_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_out_dir, exist_ok=True)
    
    # Initialize Engine
    llm, sampling_params = setup_llm()
    
    # Determine Prompt
    prompt_str = config.PROMPT # Default fallback
    if args.prompt_file:
        if os.path.exists(args.prompt_file):
            try:
                with open(args.prompt_file, 'r', encoding='utf-8') as f:
                    prompt_str = f.read().strip()
                print(f"Loaded custom prompt from: {args.prompt_file}")
            except Exception as e:
                print(f"Error reading prompt file: {e}. Using default prompt.")
        else:
            print(f"Warning: Prompt file '{args.prompt_file}' not found. Using default prompt.")
    else:
        print("Using default prompt from config.")

    # Append Schema if provided
    if args.schema_file:
        if os.path.exists(args.schema_file):
            try:
                with open(args.schema_file, 'r', encoding='utf-8') as f:
                    schema_content = f.read().strip()
                prompt_str += f"\n\nPlease extract the following metadata for each table/section, strictly adhering to this schema:\n\n{schema_content}"
                print(f"Loaded schema from: {args.schema_file}")
            except Exception as e:
                print(f"Error reading schema file: {e}")
        else:
            print(f"Warning: Schema file '{args.schema_file}' not found.")
    
    pdf_files = glob.glob(os.path.join(args.input_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")
    
    results = [] # List to store JSON results
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        print(f"Processing: {filename}")
        
        pdf_images = pdf_to_images(pdf_path)
        if not pdf_images:
            print(f"No images extracted from {filename}")
            continue
            
        # Prepare inputs
        batch_inputs = []
        for img in pdf_images:
            # Preprocess using DeepseekOCRProcessor logic
            # Note: We create a new processor instance or reuse one.
            # Original code: DeepseekOCRProcessor().tokenize_with_images(...)
            
            tokenized = DeepseekOCRProcessor().tokenize_with_images(
                images=[img], 
                bos=True, 
                eos=True, 
                cropping=config.CROP_MODE
            )
            
            batch_inputs.append({
                "prompt": prompt_str,
                "multi_modal_data": {"image": tokenized}
            })
            
        # Generate
        print(f"Generating for {len(batch_inputs)} pages...")
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        
        doc_data = {
            "file": filename,
            "pages": []
        }
        
        for idx, (output, img) in enumerate(zip(outputs, pdf_images)):
            text_content = output.outputs[0].text
            
            # Clean generic eos token if present
            text_content = text_content.replace('<｜end▁of▁sentence｜>', '')
            
            # Extract Images/Figures
            # 1. Regex Match for refs
            matches_all, matches_images, matches_other = re_match(text_content)
            
            # 2. Crop and Save Images
            saved_img_paths = crop_and_save_images(img, matches_images, images_out_dir, idx, base_name)
            
            # 3. Tables?
            # We can't easily ground tables specifically without a specific prompt OR parsing the markdown.
            # We will just verify if markdown tables exist.
            tables_found = []
            # Simple heuristic for tables in markdown
            if re.search(r'\|.*\|', text_content):
                # This is a very weak check, but indicates table existence. 
                # Improving this would require a full markdown parser.
                # For now, we return the full content, user can parse JSON.
                pass

            # 4. Clean Content (Replace image refs with local paths, remove other refs)
            cleaned_content = text_content
            
            # Replace image refs with actual paths in markdown?
            # The original code replaces matched image refs with ![](images/...)
            # We should replace with our new paths.
            
            # Recalculate img_idx for replacement alignment
            # This is tricky because crop_and_save_images iterates matches_images.
            # We need to loop again or coordinate.
            
            current_img_idx = 0
            for match_str in matches_images:
                # Construct path that was saved
                # fname = f"{base_filename}_p{idx}_{current_img_idx}.jpg"
                # Replacing match with markdown image
                if current_img_idx < len(saved_img_paths):
                     rel_path = os.path.relpath(saved_img_paths[current_img_idx], args.output_dir)
                     cleaned_content = cleaned_content.replace(match_str, f'\n![Figure]({rel_path})\n')
                current_img_idx += 1
                
            # Remove other refs
            for match_str in matches_other:
                cleaned_content = cleaned_content.replace(match_str, '')

            # Cleanup formatting
            cleaned_content = cleaned_content.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
            
            # 5. Parse and Save Artifacts (CSVs, Metadata)
            artifacts = parse_and_save_artifacts(cleaned_content, args.output_dir, idx, base_name)
            
            page_record = {
                "page": idx + 1,
                "content": cleaned_content,
                "raw_content": text_content,
                "images": [os.path.relpath(p, args.output_dir) for p in saved_img_paths],
                "csvs": artifacts["csvs"],
                "metadata": artifacts["metadata"]
            }
            doc_data["pages"].append(page_record)
            
        results.append(doc_data)
        
    # Save Output JSON
    json_path = os.path.join(args.output_dir, "output.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Processing Complete. Results saved to {json_path}")

if __name__ == "__main__":
    main()
