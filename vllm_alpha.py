"""Inference endpoint A 

Serves the Qwen 3 4B Instruct model (~4B parameters, BF16) on an RTX 4090. 
"""

from runpod_flash import Endpoint, GpuGroup, CudaVersion

vllm_alpha = Endpoint(
    name="vllm-alpha",
    image="runpod/worker-v1-vllm:v2.22.4",
    gpu=GpuGroup.ADA_24,  # RTX 4090 24GB
    workers=(0, 2), 
    min_cuda_version=CudaVersion.V13_0,
    env={
        # 3-5B instruction-tuned model served by vLLM (public, no HF token needed)
        "MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507",
        # Cap context length to limit KV-cache memory usage
        "MAX_MODEL_LEN": "8192",
        # BF16 is recommended for Qwen3; 4B model fits easily in 24GB VRAM
        "DTYPE": "bfloat16",
        # Leave a little headroom on the GPU
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_CONCURRENCY": "30",
        # Qwen3 tool-calling support (per RunPod vLLM configuration docs)
        "ENABLE_AUTO_TOOL_CHOICE": "true",
        "TOOL_CALL_PARSER": "hermes",
        # Speed up downloads with RunPod's cache acceleration
        "ACCELERATE_DOWNLOADS": "true",
    },
)