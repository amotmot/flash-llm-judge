"""Inference endpoint B

Serves Microsoft's Phi-4. A 14B-parameter text-only model.
BF16 weights (~28GB) require a 48GB GPU to leave room for KV cache.
This runs on an L40 (ADA_48_PRO).
"""

from runpod_flash import Endpoint, GpuGroup, CudaVersion

vllm_beta = Endpoint(
    name="vllm-beta",
    image="runpod/worker-v1-vllm:v2.22.4",
    gpu=GpuGroup.ADA_48_PRO,  # L40 48GB; 14B BF16 (~28GB) + KV cache
    workers=(0, 2),
    min_cuda_version=CudaVersion.V13_0,
    env={
        "MODEL_NAME": "microsoft/Phi-4",
        "MAX_MODEL_LEN": "8192",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_CONCURRENCY": "20",
        "TRUST_REMOTE_CODE": "true",
        "ACCELERATE_DOWNLOADS": "true",
    },
)