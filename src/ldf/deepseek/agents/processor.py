import os
import json
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from ldf.deepseek.agents.base import Agent

class ProcessorAgent(Agent):
    """
    Agent 2: Processor
    Role: Structure the OCR text using an LLM.
    """
    def __init__(self, llm_engine: LLM, prompt_template: str, model_path: str):
        super().__init__("Processor")
        self.llm = llm_engine
        self.prompt_template = prompt_template
        self.model_path = model_path
        self.sampling_params = SamplingParams(
            temperature=0.1, 
            max_tokens=2048, 
            stop=["```\n"],
            repetition_penalty=1.2 # Reduce hallucination
        )
        # Initialize tokenizer (lightweight, shouldn't cost VRAM)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    def process(self, extracted_pages: List[Dict[str, Any]], output_dir: Optional[str] = None, pdf_name: str = "") -> (List[Dict[str, Any]], Dict[str, int]):
        self.log("Processing extracted text with LLM...")
        
        prompts = []
        for page in extracted_pages:
            content = page['content']
            
            # Use tokenizer chat template for robustness
            messages = [
                {"role": "system", "content": "You are a precise data extraction assistant. Output only JSON."},
                {"role": "user", "content": f"Instruction:\n{self.prompt_template}\n\nInput:\n{content}\n\nResponse:"}
            ]
            
            full_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts.append(full_prompt)
            
        self.log(f"Generating structure for {len(prompts)} pages...")
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        
        # Calculate Token Usage
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        self.log(f"Processor Token Usage - Input: {total_input_tokens}, Output: {total_output_tokens}")
        usage_stats = {'input': total_input_tokens, 'output': total_output_tokens}
        
        for idx, output in enumerate(outputs):
            gen_text = output.outputs[0].text
            extracted_pages[idx]['llm_raw_output'] = gen_text
            extracted_pages[idx]['json_data'] = self._extract_json(gen_text)
            
        # Save Intermediate Artifact
        if output_dir and pdf_name:
             inter_dir = os.path.join(output_dir, "intermediate", pdf_name)
             os.makedirs(inter_dir, exist_ok=True)
             
             structured_data = [
                 {
                     "page_num": p['page_num'],
                     "llm_output": p['llm_raw_output'],
                     "json_data": p['json_data']
                 }
                 for p in extracted_pages
             ]
             
             json_path = os.path.join(inter_dir, "llm_structured_output.json")
             with open(json_path, 'w', encoding='utf-8') as f:
                 json.dump(structured_data, f, indent=2, ensure_ascii=False)
             self.log(f"Saved intermediate structured data to: {json_path}")

        return extracted_pages, usage_stats

    def _extract_json(self, text: str) -> Any:
        try:
            # 1. Try markdown code block
            start = text.find('```json')
            if start != -1:
                end = text.find('```', start + 7)
                if end != -1:
                    json_str = text[start + 7 : end].strip()
                    return json.loads(json_str)
                    
            # 2. Try generic code block
            start = text.find('```')
            if start != -1:
                end = text.find('```', start + 3)
                if end != -1:
                    json_str = text[start + 3 : end].strip()
                    try:
                        return json.loads(json_str)
                    except:
                        pass
            
            # 3. Fallback: Finding first { and last }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except:
             pass
        
        return {"error": "Could not parse JSON", "raw": text}
