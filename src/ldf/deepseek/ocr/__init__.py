from vllm.model_executor.models.registry import ModelRegistry
from .model import DeepseekOCRForCausalLM

def setup_registry():
    try:
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
    except Exception as e:
        print(f"Warning: Failed to register DeepseekOCRForCausalLM: {e}")

setup_registry()
