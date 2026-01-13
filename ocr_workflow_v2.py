import argparse
import os
import sys

# Ensure local src is in path for development
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Ensure environment is setup before importing agents
import ldf.deepseek.ocr 
from ldf.deepseek.agents.orchestrator import OrchestratorAgent

def main():
    parser = argparse.ArgumentParser(description="DeepSeek OCR Agentic Workflow V2")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing PDFs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompt text file")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-OCR", help="Path to OCR model")
    
    args = parser.parse_args()
    
    # Environment Setup for VLLM
    # We must ensure VLLM_USE_V1='0' is set in the process environment from the start.
    if os.environ.get('VLLM_USE_V1') != '0':
        print("Restarting script with VLLM_USE_V1=0 to disable V1 engine...")
        new_env = os.environ.copy()
        new_env['VLLM_USE_V1'] = '0'
        
        # Pass through other envs?
        # new_env['CUDA_VISIBLE_DEVICES'] = '0' 
        
        os.execve(sys.executable, [sys.executable] + sys.argv, new_env)

    agent = OrchestratorAgent(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        model_path=args.model_path
    )
    
    agent.run()


if __name__ == "__main__":
    main()
