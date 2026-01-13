import os
import sys
import glob
import json
import argparse
import re
import io
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# ==========================================
# Environment Setup (Must be before imports)
# ==========================================
# Ensure VLLM V1 is disabled for DeepSeek-OCR
if os.environ.get('VLLM_USE_V1') != '0':
    print("Restarting script with VLLM_USE_V1=0 to disable V1 engine...")
    new_env = os.environ.copy()
    new_env['VLLM_USE_V1'] = '0'
    new_env['CUDA_VISIBLE_DEVICES'] = '0'
    os.execve(sys.executable, [sys.executable] + sys.argv, new_env)

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Setup Path to import DeepSeek-OCR modules
current_dir = os.getcwd()
deepseek_vllm_path = os.path.join(current_dir, "external/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm")
sys.path.append(deepseek_vllm_path)

# ==========================================
# Imports
# ==========================================
try:
    import torch
    from PIL import Image
    import fitz
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry
    
    # DeepSeek Imports
    from deepseek_ocr import DeepseekOCRForCausalLM
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    from process.image_process import DeepseekOCRProcessor
    import config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Please define 'external/DeepSeek-OCR' submodule properly.")
    sys.exit(1)

# Register Model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# ==========================================
# Logic Helpers (Reused from ocr_app.py)
# ==========================================
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

def crop_and_save_images(image, matches_images, output_dir, page_idx, base_filename):
    image_width, image_height = image.size
    saved_images = []
    
    img_idx = 0
    for match in matches_images:
        try:
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

# ==========================================
# Agent Definitions
# ==========================================

class Agent:
    def __init__(self, name: str):
        self.name = name

    def log(self, message: str):
        print(f"[{self.name}] {message}")

class ExtractorAgent(Agent):
    """
    Agent 1: Extraction
    Role: Raw OCR using DeepSeek-OCR.
    Input: PDF Path
    Output: List of {page_num, raw_content, image object}
    """
    def __init__(self, llm_engine: LLM):
        super().__init__("Extractor")
        self.llm = llm_engine
        # OCR Prompt
        self.ocr_prompt = "Explain this image in detail." # Default from ocr_app config logic

        # Setup Sampling Params for OCR
        logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            logits_processors=logits_processors,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

    def process(self, pdf_path: str, output_dir: Optional[str] = None) -> (List[Dict[str, Any]], Dict[str, int]):
        self.log(f"Converting PDF to images: {pdf_path}")
        images = pdf_to_images(pdf_path)
        if not images:
            self.log("No images found.")
            return [], {'input': 0, 'output': 0}

        # Prepare Batch
        batch_inputs = []
        processor = DeepseekOCRProcessor()
        
        for img in images:
            tokenized = processor.tokenize_with_images(
                images=[img], 
                bos=True, 
                eos=True, 
                cropping=config.CROP_MODE
            )
            batch_inputs.append({
                "prompt": self.ocr_prompt,
                "multi_modal_data": {"image": tokenized}
            })

        self.log(f"Running OCR on {len(batch_inputs)} pages...")
        outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
        
        # Calculate Token Usage
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        self.log(f"OCR Token Usage - Input: {total_input_tokens}, Output: {total_output_tokens}")
        usage_stats = {'input': total_input_tokens, 'output': total_output_tokens}
        
        results = []
        intermediate_data = [] # For JSON serialization
        
        # Setup Intermediate Directory if output_dir provided
        save_intermediate = output_dir is not None
        images_out_dir = None
        if save_intermediate:
            base_name = Path(pdf_path).stem
            inter_dir = os.path.join(output_dir, "intermediate", base_name)
            images_out_dir = os.path.join(inter_dir, "images")
            os.makedirs(images_out_dir, exist_ok=True)
        
        for idx, (output, img) in enumerate(zip(outputs, images)):
            text = output.outputs[0].text
            # Clean generic eos token
            text = text.replace('<｜end▁of▁sentence｜>', '')
            
            # 1. Parse Image Refs
            matches_all, matches_images, matches_other = re_match(text)
            
            # 2. Crop and Save Images (if intermediate saving enabled)
            saved_img_paths = []
            if save_intermediate and images_out_dir:
                # Also save the full page for reference
                full_page_path = os.path.join(images_out_dir, f"{base_name}_p{idx+1}_full.png")
                img.save(full_page_path)
                
                # Crop detections
                saved_img_paths = crop_and_save_images(img, matches_images, images_out_dir, idx, base_name)
            
            # 3. Create Cleaned Content (Replace refs with markdown)
            cleaned_content = text
            current_img_idx = 0
            for match_str in matches_images:
                if current_img_idx < len(saved_img_paths) and save_intermediate:
                     rel_path = os.path.relpath(saved_img_paths[current_img_idx], output_dir)
                     cleaned_content = cleaned_content.replace(match_str, f'\n![Figure]({rel_path})\n')
                else:
                     # If not saving intermediate, strictly we can't link images easily.
                     # Just remove or leave placeholder? User wants content. 
                     # Let's remove if we can't link, or keep placeholder.
                     # "content: cleaner version". 
                     cleaned_content = cleaned_content.replace(match_str, '')
                current_img_idx += 1
            
            # Remove other refs
            for match_str in matches_other:
                cleaned_content = cleaned_content.replace(match_str, '')

            # Cleanup formatting
            cleaned_content = cleaned_content.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

            results.append({
                "page_num": idx + 1,
                "raw_content": text,
                "content": cleaned_content, # New field
                "image": img
            })
            
            intermediate_data.append({
                "page_num": idx + 1,
                "raw_content": text,
                "content": cleaned_content,
                "image_paths": [os.path.relpath(p, output_dir) for p in saved_img_paths] if save_intermediate else []
            })
            
        # Save Intermediate JSON
        if save_intermediate:
            json_path = os.path.join(inter_dir, "ocr_raw_output.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
            self.log(f"Saved intermediate OCR results to: {json_path}")
            
        return results, usage_stats

class ProcessorAgent(Agent):
    """
    Agent 2: Processing
    Role: Structure the Raw OCR content using the User's Prompt.
    Input: List of Raw Page Data
    Output: List of Structured Page Data (JSON)
    """
    def __init__(self, llm_engine: LLM, user_prompt_template: str):
        super().__init__("Processor")
        self.llm = llm_engine
        self.user_prompt = user_prompt_template
        # Stricter sampling for JSON
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=4096,
            skip_special_tokens=True, # We want clean JSON
            repetition_penalty=1.2 # Prevent looping
        )

    def process(self, raw_pages: List[Dict[str, Any]], output_dir: Optional[str] = None, pdf_name: str = "") -> (List[Dict[str, Any]], Dict[str, int]):
        self.log("Structuring data using User Prompt...")
        
    def process(self, raw_pages: List[Dict[str, Any]], output_dir: Optional[str] = None, pdf_name: str = "") -> List[Dict[str, Any]]:
        self.log("Structuring data using User Prompt...")
        
        # Get Tokenizer for Chat Template
        tokenizer = self.llm.get_tokenizer()
        
        final_prompts = []
        
        for page in raw_pages:
            content = page.get('content', page['raw_content'])
            
            # Construct Messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant that extracts structured data tables from text. Output ONLY valid JSON."},
                {"role": "user", "content": f"{self.user_prompt}\n\n--- INPUT DATA (Page {page['page_num']}) ---\n{content}\n\n--- END INPUT ---\n\nRESPONSE (JSON ONLY):"}
            ]
            
            # Apply Template
            # This handles <|im_start|>, [INST], etc. automatically based on the loaded model
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                # Fallback if no chat template exists (e.g. base model)
                self.log(f"Warning: Could not apply chat template ({e}). Using raw concatenation.")
                formatted_prompt = f"{self.user_prompt}\n\nInput:\n{content}\n\nOutput JSON:"

            final_prompts.append(formatted_prompt)

        # Run LLM
        outputs = self.llm.generate(final_prompts, sampling_params=self.sampling_params)

        outputs = self.llm.generate(final_prompts, sampling_params=self.sampling_params)
        
        # Calculate Token Usage
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        self.log(f"Processor Token Usage - Input: {total_input_tokens}, Output: {total_output_tokens}")
        usage_stats = {'input': total_input_tokens, 'output': total_output_tokens}
        
        processed_pages = []
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text
            
            # Extract JSON block
            json_data = self._extract_json(response_text)
            
            processed_pages.append({
                "page_num": raw_pages[i]['page_num'],
                "json_data": json_data,
                "raw_response": response_text
            })
            
        # Save Intermediate Artifacts
        if output_dir and pdf_name:
            inter_dir = os.path.join(output_dir, "intermediate", pdf_name)
            os.makedirs(inter_dir, exist_ok=True)
            
            json_path = os.path.join(inter_dir, "llm_structured_output.json")
            
            # Create a serializable version (remove non-serializable if any, though dicts should be fine)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(processed_pages, f, indent=2, ensure_ascii=False)
            self.log(f"Saved intermediate Processor results to: {json_path}")
            
        return processed_pages, usage_stats
    
    def _extract_json(self, text):
        # basic regex extraction
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if match:
             try:
                 return json.loads(match.group(1))
             except:
                 pass
        
        # Try without code blocks if it looks like JSON
        try:
             stripped = text.strip()
             if (stripped.startswith('{') or stripped.startswith('[')) and (stripped.endswith('}') or stripped.endswith(']')):
                 return json.loads(stripped)
        except:
             pass

        # Fallback: try finding first { and last }
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except:
             pass
        
        return {"error": "Could not parse JSON", "raw": text}

class AggregatorAgent(Agent):
    """
    Agent 3: Aggregator
    Role: Stitch pages based on 'CONTINUATION' logic.
    """
    def __init__(self):
        super().__init__("Aggregator")

    def process(self, processed_pages: List[Dict[str, Any]], output_dir: Optional[str] = None, pdf_name: str = "") -> List[Dict[str, Any]]:
        self.log("Aggregating pages...")
        
        consolidated_ministers = []
        current_minister = None
        debug_log = [] # detailed log of decisions
        
        for page in processed_pages:
            data = page.get('json_data', {})
            page_num = page['page_num']
            
            # Normalize data to list
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                if not isinstance(item, dict): continue
                
                raw_name = item.get("Minister") or ""
                # Use extracted name directly for the record
                minister_name = raw_name.strip()
                
                # Normalization for Comparison Logic (Internal Use Only)
                norm_name = minister_name.upper().replace("HON.", "").replace("DR.", "").replace("MR.", "").strip()
                # Remove common continuation markers for comparison
                norm_name_clean = norm_name.replace("(CONTD", "").replace("(CONTINUED", "").replace("CONTD", "").replace("CONTINUED", "").strip(" .():")
                
                # Check current minister for comparison
                current_norm = ""
                if current_minister:
                    curr_raw = current_minister.get("Minister", "")
                    current_norm = curr_raw.upper().replace("HON.", "").replace("DR.", "").strip()
                    current_norm = current_norm.replace("(CONTD", "").replace("(CONTINUED", "").replace("CONTD", "").replace("CONTINUED", "").strip(" .():")
                
                is_continuation_keyword = (
                    minister_name == "CONTINUATION_FROM_PREVIOUS" or 
                    "CONTINUATION" in minister_name.upper() or
                    "(CONTD" in minister_name.upper() or
                    "(CONTINUED" in minister_name.upper()
                )
                
                # MERGE IF: Explicit Keyword OR Name Matches Previous (ignoring suffixes)
                should_merge = False
                if is_continuation_keyword:
                    should_merge = True
                elif current_minister and norm_name_clean and (norm_name_clean == current_norm):
                    should_merge = True
                    self.log(f"Detected name match for merge: '{minister_name}' == '{curr_raw}'")

                decision = "UNKNOWN"
                
                if should_merge:
                    if current_minister:
                        decision = "MERGED"
                        self.log(f"Merging continuation on Page {page_num} to {current_minister.get('Minister')}")
                        # Merge Logic
                        for col in ["Column I", "Column II", "Column III"]:
                            if col in item:
                                # Ensure we have target list
                                if col not in current_minister: current_minister[col] = []
                                
                                # Get new values
                                new_vals = item[col]
                                if isinstance(new_vals, str): new_vals = [new_vals]
                                if not isinstance(new_vals, list): new_vals = []
                                
                                current_minister[col].extend(new_vals)
                    else:
                        decision = "ORPHANED_CONTINUATION"
                        self.log(f"Warning: Orphaned continuation on Page {page_num}")
                        
                else:
                    # New Minister
                    if minister_name:
                        decision = "NEW_MINISTER"
                        current_minister = item.copy() # Start new
                        current_minister['Minister'] = minister_name # Keep raw name
                        current_minister['_source_pages'] = [page_num]
                        
                        # Ensure columns become lists if they are strings
                        for col in ["Column I", "Column II", "Column III"]:
                            val = current_minister.get(col, [])
                            if isinstance(val, str): current_minister[col] = [val]
                            
                        consolidated_ministers.append(current_minister)
                    else:
                        decision = "SKIPPED_EMPTY_NAME"
                        
                debug_log.append({
                    "page": page_num,
                    "raw_minister": raw_name,
                    "norm_name": norm_name_clean,
                    "decision": decision,
                    "target_minister": current_minister.get("Minister") if current_minister else None
                })

        # Save Debug Artifact
        if output_dir and pdf_name:
             inter_dir = os.path.join(output_dir, "intermediate", pdf_name)
             os.makedirs(inter_dir, exist_ok=True)
             debug_path = os.path.join(inter_dir, "aggregator_debug_log.json")
             with open(debug_path, 'w', encoding='utf-8') as f:
                 json.dump(debug_log, f, indent=2, ensure_ascii=False)
             self.log(f"Saved Aggregator debug log to: {debug_path}")
                         
        return consolidated_ministers

class FinalizerAgent(Agent):
    """
    Agent 4: Finalizer
    Role: Save output.
    """
    def __init__(self):
        super().__init__("Finalizer")

    def process(self, data: Any, output_dir: str):
        self.log(f"Saving final output to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save JSON
        out_path_json = os.path.join(output_dir, "final_consolidated_output.json")
        with open(out_path_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.log(f"Saved JSON: {out_path_json}")
        
        # 2. Save CSV (Flattened)
        out_path_csv = os.path.join(output_dir, "final_consolidated_output.csv")
        try:
            import csv
            
            # Determine max columns dynamically or fixed?
            # Prompt asked for Column I, II, III.
            headers = ["Minister", "Row_ID", "Column I", "Column II", "Column III"]
            
            with open(out_path_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for entry in data:
                    minister = entry.get("Minister", "Unknown")
                    
                    # Align columns by index
                    col1 = entry.get("Column I", [])
                    col2 = entry.get("Column II", [])
                    col3 = entry.get("Column III", [])
                    
                    # Ensure they are lists
                    if isinstance(col1, str): col1 = [col1]
                    if isinstance(col2, str): col2 = [col2]
                    if isinstance(col3, str): col3 = [col3]
                    
                    # Get max length to iterate rows
                    max_len = max(len(col1), len(col2), len(col3))
                    
                    for i in range(max_len):
                        val1 = col1[i] if i < len(col1) else ""
                        val2 = col2[i] if i < len(col2) else ""
                        val3 = col3[i] if i < len(col3) else ""
                        
                        writer.writerow([minister, i+1, val1, val2, val3])
                        
            self.log(f"Saved CSV: {out_path_csv}")
            
        except Exception as e:
            self.log(f"Error saving CSV: {e}")

        # 3. Create Per-Minister Folders
        self.log("Creating per-minister output folders...")
        try:
            for entry in data:
                minister_name = entry.get("Minister", "Unknown")
                
                # Snake Case Conversion
                # 1. Remove titles (Hon., etc) for folder name clarity
                clean_for_folder = minister_name.replace("Hon.", "").replace("Dr.", "").replace("Mr.", "")
                # 2. Lowercase and replace spaces/symbols with _
                safe_name = "".join([c if c.isalnum() else '_' for c in clean_for_folder]).strip('_')
                # 3. Collapse multiple underscores
                while "__" in safe_name:
                    safe_name = safe_name.replace("__", "_")
                safe_name = safe_name.lower()
                
                if not safe_name: safe_name = "unknown_minister"
                
                minister_dir = os.path.join(output_dir, safe_name)
                os.makedirs(minister_dir, exist_ok=True)
                
                # A. Save Minister JSON
                min_json_path = os.path.join(minister_dir, "data.json")
                with open(min_json_path, 'w', encoding='utf-8') as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
                    
                # B. Save Minister CSV
                min_csv_path = os.path.join(minister_dir, "data.csv")
                headers = ["Minister", "Row_ID", "Column I", "Column II", "Column III"]
                
                with open(min_csv_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    
                    col1 = entry.get("Column I", [])
                    col2 = entry.get("Column II", [])
                    col3 = entry.get("Column III", [])
                    
                    if isinstance(col1, str): col1 = [col1]
                    if isinstance(col2, str): col2 = [col2]
                    if isinstance(col3, str): col3 = [col3]
                    
                    max_len = max(len(col1), len(col2), len(col3))
                    
                    for i in range(max_len):
                        val1 = col1[i] if i < len(col1) else ""
                        val2 = col2[i] if i < len(col2) else ""
                        val3 = col3[i] if i < len(col3) else ""
                        writer.writerow([minister_name, i+1, val1, val2, val3])
                        
        except Exception as e:
             self.log(f"Error creating minister folders: {e}")

# ==========================================
# Orchestrator
# ==========================================
# ==========================================
# Orchestrator
# ==========================================
class OrchestratorAgent(Agent):
    def __init__(self, args):
        super().__init__("Orchestrator")
        self.args = args
        
        # Load User Prompt
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            self.user_prompt_content = f.read()

        self.aggregator = AggregatorAgent()
        self.finalizer = FinalizerAgent()

    def run(self):
        import time
        start_time = time.time()
        
        # Token Counters
        workflow_input_tokens = 0
        workflow_output_tokens = 0

        # Scan Inputs
        input_path = Path(self.args.input_dir)
        pdf_files = list(input_path.glob("*.pdf"))
        self.log(f"Found {len(pdf_files)} PDF files.")
        
        # --- PHASE 1: EXTRACTION (OCR MODEL) ---
        self.log("--- PHASE 1: STARTING EXTRACTION (OCR) ---")
        phase1_start = time.time()
        ocr_model_path = self.args.model_path if self.args.model_path else config.MODEL_PATH
        self.log(f"Loading OCR Model: {ocr_model_path}")
        
        # Initialize OCR Engine
        ocr_llm = LLM(
            model=ocr_model_path,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            trust_remote_code=True,
            max_model_len=8192,
            max_num_seqs=config.MAX_CONCURRENCY,
            gpu_memory_utilization=0.9,
            disable_mm_preprocessor_cache=True
        )
        
        extractor = ExtractorAgent(ocr_llm)
        
        # Initializing storage for raw results to pass to Phase 2
        # Map: pdf_path (str) -> raw_pages (List[Dict])
        all_raw_pages = {} 
        
        for pdf in pdf_files:
            try:
                self.log(f"Extracting: {pdf.name}")
                raw_pages, usage = extractor.process(str(pdf), self.args.output_dir)
                all_raw_pages[str(pdf)] = raw_pages
                
                workflow_input_tokens += usage.get('input', 0)
                workflow_output_tokens += usage.get('output', 0)
            except Exception as e:
                self.log(f"Error extracting {pdf.name}: {e}")
                import traceback
                traceback.print_exc()

        # UNLOAD OCR MODEL
        self.log("Unloading OCR Model to free VRAM...")
        del extractor
        del ocr_llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.log("OCR Model Unloaded.")
        phase1_duration = time.time() - phase1_start
        self.log(f"Phase 1 (OCR) Duration: {phase1_duration:.2f} seconds")
        
        # --- PHASE 2: PROCESSING (TEXT MODEL) ---
        self.log("--- PHASE 2: STARTING PROCESSING (TEXT STRUCTURING) ---")
        phase2_start = time.time()
        
        # Hardcoded default as per requirements
        processor_model_path = "Qwen/Qwen2.5-7B-Instruct" 
                    
        self.log(f"Loading Processor Model: {processor_model_path}")
        
        # Recalculate tokenizer settings or specific configs for Text Model?
        # VLLM handles auto config usually.
        # Note: If reusing DeepSeek-OCR for text, we need the override. If using Qwen/Llama, we DO NOT.
        
        processed_hf_overrides = None
        if "deepseek-ocr" in processor_model_path.lower():
             processed_hf_overrides = {"architectures": ["DeepseekOCRForCausalLM"]}
             
        processor_llm = LLM(
            model=processor_model_path,
            hf_overrides=processed_hf_overrides, 
            trust_remote_code=True,
            max_model_len=8192, # Adjust based on model capability
            max_num_seqs=config.MAX_CONCURRENCY,
            gpu_memory_utilization=0.9
        )
        
        processor = ProcessorAgent(processor_llm, self.user_prompt_content)
        
        for pdf in pdf_files:
            pdf_str = str(pdf)
            if pdf_str not in all_raw_pages:
                continue
                
            try:
                self.log(f"Processing: {pdf.name}")
                raw_pages = all_raw_pages[pdf_str]
                
                # Step 2: Process
                structured_pages, usage = processor.process(
                    raw_pages, 
                    output_dir=self.args.output_dir,
                    pdf_name=pdf.stem
                )
                
                workflow_input_tokens += usage.get('input', 0)
                workflow_output_tokens += usage.get('output', 0)
                
                # Step 3: Aggregate
                consolidated_data = self.aggregator.process(
                    structured_pages, 
                    output_dir=self.args.output_dir,
                    pdf_name=pdf.stem
                )
                
                # Step 4: Finalize
                output_subdir = os.path.join(self.args.output_dir, pdf.stem)
                self.finalizer.process(consolidated_data, output_subdir)
                
                self.log(f"--- Completed Workflow for {pdf.name} ---")
                
            except Exception as e:
                self.log(f"Error processing {pdf.name}: {e}")
                import traceback
                traceback.print_exc()
        
        
        phase2_duration = time.time() - phase2_start
        total_duration = time.time() - start_time
        
        self.log(f"Phase 2 (Processing) Duration: {phase2_duration:.2f} seconds")
        self.log(f"Workflow Complete. Total Time: {total_duration:.2f} seconds")
        self.log(f"Total Workflow Token Usage - Input: {workflow_input_tokens}, Output: {workflow_output_tokens}")

        # Save Metrics to Disk
        metrics = {
            "total_duration_seconds": total_duration,
            "phase1_duration_seconds": phase1_duration,
            "phase2_duration_seconds": phase2_duration,
            "token_usage": {
                "total_input": workflow_input_tokens,
                "total_output": workflow_output_tokens
            }
        }
        metrics_path = os.path.join(self.args.output_dir, "workflow_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        self.log(f"Saved workflow metrics to: {metrics_path}")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek Multi-Agent OCR Workflow")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    
    args = parser.parse_args()
    
    orchestrator = OrchestratorAgent(args)
    orchestrator.run()
