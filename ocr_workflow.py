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
sys.path.insert(0, deepseek_vllm_path)

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

    def process(self, pdf_path: str) -> List[Dict[str, Any]]:
        self.log(f"Converting PDF to images: {pdf_path}")
        images = pdf_to_images(pdf_path)
        if not images:
            self.log("No images found.")
            return []

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
        
        results = []
        for idx, (output, img) in enumerate(zip(outputs, images)):
            text = output.outputs[0].text
            # Clean generic eos token
            text = text.replace('<｜end▁of▁sentence｜>', '')
            
            results.append({
                "page_num": idx + 1,
                "raw_content": text,
                "image": img # Keep image in memory if needed for crops, or just drop it
            })
            
        return results

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
            skip_special_tokens=True # We want clean JSON
        )

    def process(self, raw_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.log("Structuring data using User Prompt...")
        
        batch_prompts = []
        
        for page in raw_pages:
            raw_text = page['raw_content']
            
            # Construct Prompt
            # We wrap the OCR text effectively as context
            full_prompt = (
                f"{self.user_prompt}\n\n"
                f"Here is the raw OCR content from Page {page['page_num']}:\n"
                f"```text\n{raw_text}\n```\n\n"
                f"Please output ONLY the JSON structure as requested."
            )
            
            # For DeepSeek-VL-Chat (or OCR model behaving as VLM), we pass text prompt.
            # vllm.generate accepts list of strings for text-only generation.
            batch_inputs = [full_prompt] # Wait, batch_inputs for generate is list of prompts.
             
            # Actually we collect all pages to run one batch?
            # Yes, let's process all pages in one batch for efficiency.
            batch_prompts.append(full_prompt)

        # Run LLM
        # Note: Depending on the model, we might need chat formatting.
        # But let's try raw prompt injection first as DeepSeek usually handles it.
        # If the model is strictly instruction tuned with a template, we might need <|User|> ...
        
        # DeepSeek-VL Chat Template usually:
        # <|User|>: ... <|Assistant|>:
        # Let's apply a simple check or default wrapper.
        final_prompts = []
        for p in batch_prompts:
            # Simple manual chat template approximation if not using tokenizer.apply_chat_template
            final_prompts.append(f"<|User|>: {p}\n<|Assistant|>:")

        outputs = self.llm.generate(final_prompts, sampling_params=self.sampling_params)
        
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
            
        return processed_pages

    def _extract_json(self, text):
        # basic regex extraction
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if match:
             try:
                 return json.loads(match.group(1))
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

    def process(self, processed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.log("Aggregating pages...")
        
        consolidated_ministers = []
        current_minister = None
        
        for page in processed_pages:
            data = page.get('json_data', {})
            
            # Handle list of items or single item?
            # Prompt says: "Minister": "Name", "Column I": [...]
            # It returns ONE object per page usually? Or multiple?
            # The prompt implies 3 objects? No, "Output data in JSON format where each minister will have 3 objects." (Confusing prompt text in memory)
            # Actually prompt says: "format... { Minister: ..., Column I: ... }"
            
            # Let's assume data is a Dict or List of Dicts.
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                if not isinstance(item, dict): continue
                
                minister_name = item.get("Minister", "")
                
                is_continuation = (
                    minister_name == "CONTINUATION_FROM_PREVIOUS" or 
                    "CONTINUATION" in minister_name.upper()
                )
                
                if is_continuation and current_minister:
                    # Merge data
                    self.log(f"Merging continuation on Page {page['page_num']} to {current_minister.get('Minister')}")
                    for col in ["Column I", "Column II", "Column III"]:
                        if col in item and col in current_minister:
                            # Assuming list of strings
                            if isinstance(item[col], list) and isinstance(current_minister[col], list):
                                current_minister[col].extend(item[col])
                else:
                    # New Minister
                    if minister_name and not is_continuation:
                        current_minister = item.copy() # Start new
                        current_minister['_source_pages'] = [page['page_num']]
                        consolidated_ministers.append(current_minister)
                    elif is_continuation and not current_minister:
                         self.log(f"Warning: Orphaned continuation on Page {page['page_num']}")
                         
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
        
        out_path = os.path.join(output_dir, "final_consolidated_output.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.log(f"Saved: {out_path}")

# ==========================================
# Orchestrator
# ==========================================
class OrchestratorAgent(Agent):
    def __init__(self, args):
        super().__init__("Orchestrator")
        self.args = args
        
        # Init LLM Engine (Shared Resource)
        self.log("Initializing Shared VLLM Engine...")
        model_path = args.model_path if args.model_path else config.MODEL_PATH
        
        self.llm = LLM(
            model=model_path,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256, # Consistent with ocr_app
            trust_remote_code=True,
            max_model_len=8192,
            max_num_seqs=config.MAX_CONCURRENCY,
            gpu_memory_utilization=0.9,
            disable_mm_preprocessor_cache=True
        )
        
        # Load User Prompt
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            user_prompt = f.read()
            
        # Init Agents
        self.extractor = ExtractorAgent(self.llm)
        self.processor = ProcessorAgent(self.llm, user_prompt)
        self.aggregator = AggregatorAgent()
        self.finalizer = FinalizerAgent()

    def run(self):
        # Scan Inputs
        input_path = Path(self.args.input_dir)
        pdf_files = list(input_path.glob("*.pdf"))
        self.log(f"Found {len(pdf_files)} PDF files.")
        
        for pdf in pdf_files:
            try:
                self.log(f"--- Starting Workflow for {pdf.name} ---")
                
                # Step 1: Extract
                raw_pages = self.extractor.process(str(pdf))
                
                # Step 2: Process
                structured_pages = self.processor.process(raw_pages)
                
                # Step 3: Aggregate
                consolidated_data = self.aggregator.process(structured_pages)
                
                # Step 4: Finalize
                output_subdir = os.path.join(self.args.output_dir, pdf.stem)
                self.finalizer.process(consolidated_data, output_subdir)
                
                self.log(f"--- Completed Workflow for {pdf.name} ---")
                
            except Exception as e:
                self.log(f"Error processing {pdf.name}: {e}")
                import traceback
                traceback.print_exc()

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
