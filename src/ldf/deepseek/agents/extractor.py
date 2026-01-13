import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from vllm import LLM, SamplingParams

# Internal Imports
from ldf.deepseek.agents.base import Agent
from ldf.deepseek.ocr.utils import pdf_to_images, re_match, crop_and_save_images
# We need these from the deepseek submodule...
from ldf.deepseek.ocr.logits_process import NoRepeatNGramLogitsProcessor
from ldf.deepseek.ocr.process import DeepseekOCRProcessor
from ldf.deepseek.config import PROMPT
# How to handle config? We might need to pass config or import it.
# For now, hardcode or accept as init params.

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
        self.ocr_prompt = PROMPT 

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
        
        # Hardcoded crop mode 'd' from original config for now
        crop_mode = 'd' 
        
        for img in images:
            tokenized = processor.tokenize_with_images(
                images=[img], 
                bos=True, 
                eos=True, 
                cropping=crop_mode
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
