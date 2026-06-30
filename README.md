# Runpod Flash LLM-as-a-Judge

Two vLLM inference endpoints serve different models. A third, larger vLLM
endpoint acts as an **LLM judge**: it receives both candidate answers and
decides which is more accurate. A lightweight CPU orchestrator coordinates
the fan-out and the judge call.

## Architecture

```
            ┌──────────────────────────────────────────────┐
            │           judge-orchestrator (CPU)           │
            └──────────┬───────────────┬────────────┬──────┘
                       │               │             │
        ┌──────────────▼──┐  ┌─────────▼──────┐  ┌───▼────────────┐
        │   vllm-alpha    │  │   vllm-beta    │  │  vllm-judge    │
        │   Qwen3-4B-AWQ  │  │  Phi-4 14B     │  │ Qwen3.6-27B-FP8│
        │   RTX 4090 24GB │  │  L40 48GB      │  │  A100 80GB     │
        │   workers 0..1  │  │  workers 0..1  │  │  workers 0..1  │
        └─────────────────┘  └────────────────┘  └────────────────┘
```

| Endpoint            |  Hardware     | Model                     | Role        |
|---------------------|-------------|---------------------------|---------------|
| `vllm-alpha`        |  ADA_24     |Qwen/Qwen3-4B-Instruct-2507| Candidate A   |
| `vllm-beta`         |  ADA_48_PRO | microsoft/Phi-4           | Candidate B   |
| `vllm-judge`        |  AMPERE_80  | Qwen/Qwen3.6-27B-FP8      | Judge         |
| `judge-orchestrator`|  cpu3c-2-4  | m                         | Coordinator   |

A judge should be **at least as capable** as the models it evaluates.

> **Note: GPUs and models are configurable. The table above is for demo purposes.**
> The specific GPUs and models shown are illustrative defaults chosen to demonstrate
> the pattern (a small candidate, a mid-size candidate, and a larger judge). Each
> endpoint is independently configurable in its worker file (`vllm_alpha.py`,
> `vllm_beta.py`, `vllm_judge.py`):
>
> - **GPU** set via the `gpu=GpuGroup.*` argument (e.g. `ADA_24`, `ADA_48_PRO`,
>   `AMPERE_80`). Pick the smallest GPU that fits the model's weights + KV cache.
> - **Model** set via the `MODEL_NAME` env var (any vLLM-compatible HF model).
>   Gated models additionally need an `HF_TOKEN`.
> - **Serving knobs** `MAX_MODEL_LEN`, `DTYPE`, `GPU_MEMORY_UTILIZATION`,
>
> Swap in whatever candidates you want to compare and whatever judge fits your
> budget. Ensure the judge is at least as capable as the candidates, and size
> the GPU to the model.

## How the queue-based flow works

The two inference endpoints and the judge all leverage pre-built
`worker-v1-vllm` image avaiable on Docker Hub. Each
job is submitted to a queue, a worker picks it up, runs generation, and
returns the result.

The orchestrator  ...

1. Receives the prompt as a job on its own queue.
2. Submits two **parallel** jobs to `vllm-alpha` and `vllm-beta` via
   `await Endpoint.runsync(...)` using `asyncio.gather` — both GPUs work at the
   same time, so wall-clock is ~max(alpha, beta)
3. Once both return, submits one job to `vllm-judge` with the prompt and
   both candidate answers.
4. Returns the assembled result to its caller.

### Cold-start behavior

All three GPU endpoints use `workers=(0, 2)`. They scale to zero when
idle. The **first** evaluation will trigger three cold starts (model
download + load into VRAM). On an A100 with accelerated downloads, the
27B FP8 judge cold-starts in ~60-90s; the 4B candidate in ~15-30s;
the 14B BF16 Phi-4 in ~30-60s. Subsequent evaluations hit warm workers
and complete in seconds.

## Deploy

```bash
cd flash-llm-judge
flash login        # once
flash deploy
```

Find endpoint IDs after deploy:

```bash
flash env get production
```

Then set the env vars on the orchestrator endpoint (via RunPod console
or re-deploy with the IDs in the orchestrator's `env=`):

```
VLLM_ALPHA_ENDPOINT_ID=<id from flash env get>
VLLM_BETA_ENDPOINT_ID=<id from flash env get>
VLLM_JUDGE_ENDPOINT_ID=<id from flash env get>
RUNPOD_API_KEY=<your api key>
```

For local testing, export them in your shell instead.

## Run

### Single prompt (direct)

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=<judge-orchestrator endpoint id>
python orchestrator.py "What is a GPU?"
```

### Batch evaluation with file output

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=<judge-orchestrator endpoint id>
python eval.py --out results.jsonl
# or with a custom prompts file:
python eval.py --prompts-file my_prompts.jsonl --out results.jsonl
```

Results are **streamed to JSONL**. Each evaluation is written and flushed
immediately, so a crash mid-batch doesn't lose completed work.
