import os
os.environ["VLLM_ATTENTION_BACKEND"]="FLASHINFER"

from vllm import LLM
from transformers import AutoTokenizer

_ENGINE_CACHE = {}

class vLLM:
    def __new__(cls, args, model_dir, dtype="auto"):
        key = (model_dir, dtype)

        if key in _ENGINE_CACHE:
            return _ENGINE_CACHE[key]

        obj = super().__new__(cls)
        _ENGINE_CACHE[key] = obj
        return obj

    def __init__(self, args, model_dir, dtype="auto"):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.name = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Qwen3.5 hybrid Mamba/GDN arch crashes CUDAGraph capture in vLLM 0.17.0
        enforce_eager = "qwen3.5" in model_dir.lower() or "qwen3_5" in model_dir.lower()

        self.llm = LLM(
            model=model_dir,
            trust_remote_code=True,
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.8),
            seed=args.seed,
            dtype=dtype,
            enforce_eager=enforce_eager,
        )