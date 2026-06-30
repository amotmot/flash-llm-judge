"""Judge endpoint

Serves Qwen3.6-27B in FP8 (~28GB weights)on an A100 80GB.
The orchestrator sends it the input prompt plus both candidate
responses and asks it to pick the more accurate answer.

At 27B parameters, this judge is larger than both candidates (Qwen3-4B and
Phi-4 14B), giving it the capacity to reliably detect factual errors and
avoid verbosity bias.

Qwen3.6 thinks by default (outputs `<think>...</think>` blocks before the
answer). The orchestrator's _parse_verdict handles this by stripping
non-JSON content and extracting the verdict via regex fallback.
"""

from runpod_flash import Endpoint, GpuGroup, CudaVersion

vllm_judge = Endpoint(
    name="vllm-judge",
    image="runpod/worker-v1-vllm:v2.22.4",
    gpu=GpuGroup.AMPERE_80,  # A100 80GB; 27B FP8 (~28GB) + KV cache
    workers=(0, 2),
    min_cuda_version=CudaVersion.V13_0,
    env={
        # Qwen3.6-27B FP8 — fits on a single 40GB+ GPU
        "MODEL_NAME": "Qwen/Qwen3.6-27B-FP8",
        "MAX_MODEL_LEN": "16384",  # cap from 262K for KV-cache memory
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_CONCURRENCY": "10",
        # Qwen3.6 reasoning parser (enables thinking mode)
        "REASONING_PARSER": "qwen3",
        # Text-only: skip vision encoder + multimodal profiling
        "LANGUAGE_MODEL_ONLY": "true",
        "ACCELERATE_DOWNLOADS": "true",
        "OPENAI_SERVED_MODEL_NAME_OVERRIDE": "vllm-judge",
    },
)