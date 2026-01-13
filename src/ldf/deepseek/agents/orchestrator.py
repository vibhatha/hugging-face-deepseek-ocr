import os
import gc
import torch
from pathlib import Path
from typing import Optional
from vllm import LLM

from ldf.deepseek.agents.base import Agent
from ldf.deepseek.agents.extractor import ExtractorAgent, DeepseekOCRProcessor
from ldf.deepseek.agents.processor import ProcessorAgent
from ldf.deepseek.agents.aggregator import AggregatorAgent
from ldf.deepseek.agents.finalizer import FinalizerAgent
# We need to register the model for VLLM
# We need to register the model for VLLM
import ldf.deepseek.ocr
 

class OrchestratorAgent(Agent):
    def __init__(self, input_dir, output_dir, prompt_file, model_path, processor_model_path="Qwen/Qwen2.5-7B-Instruct"):
        super().__init__("Orchestrator")
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.prompt_file = prompt_file
        self.model_path = model_path
        self.processor_model_path = processor_model_path
        
        # Load User Prompt
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.user_prompt_content = f.read()

        # Agents (initialized on demand or here)
        self.aggregator = AggregatorAgent()
        self.finalizer = FinalizerAgent()

    def run(self):
        # Scan Inputs
        input_path = Path(self.input_dir)
        pdf_files = list(input_path.glob("*.pdf"))
        self.log(f"Found {len(pdf_files)} PDF files.")
        
        # --- PHASE 1: EXTRACTION (OCR MODEL) ---
        self.log("--- PHASE 1: STARTING EXTRACTION (OCR) ---")
        
        # Initialize OCR Engine
        ocr_llm = LLM(
            model=self.model_path,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            trust_remote_code=True,
            max_model_len=8192,
            max_num_seqs=10, # Configurable?
            enforce_eager=True # Optimization
        )
        
        extractor = ExtractorAgent(ocr_llm)
        
        all_workflow_data = {} # Map pdf_name -> extracted_pages
        
        for pdf_file in pdf_files:
            self.log(f"Processing PDF (OCR): {pdf_file.name}")
            pages_data = extractor.process(str(pdf_file), self.output_dir)
            all_workflow_data[pdf_file.name] = pages_data
            
        # Clean up OCR Model
        del extractor
        del ocr_llm
        gc.collect()
        torch.cuda.empty_cache()
        self.log("OCR Model Unloaded.")
        
        # --- PHASE 2: PROCESSING (TEXT MODEL) ---
        self.log("--- PHASE 2: STARTING PROCESSING (TEXT) ---")
        
        # Initialize Text Engine
        text_llm = LLM(
            model=self.processor_model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.6 # Save some room?
        )
        
        processor = ProcessorAgent(text_llm, self.user_prompt_content, self.processor_model_path)
        
        for pdf_name, pages_data in all_workflow_data.items():
            self.log(f"Processing PDF (Struct): {pdf_name}")
            
            # 1. Structure (Processor)
            structured_pages = processor.process(pages_data, self.output_dir, pdf_name)
            
            # 2. Aggregate
            consolidated_data = self.aggregator.process(structured_pages, self.output_dir, pdf_name)
            
            # 3. Finalize
            # Create subfolder for this PDF's result? Or root output?
            # Existing workflow puts it in output_dir/pdf_name_no_ext usually or flat?
            # Original script did: os.path.join(args.output_dir, pdf_name_no_ext)
            
            pdf_out_dir = os.path.join(self.output_dir, Path(pdf_name).stem)
            self.finalizer.process(consolidated_data, pdf_out_dir)
            
        self.log("Workflow Complete.")
