"""Judge orchestrator: CPU endpoint that coordinates the LLM-judge flow.

Pattern A (queue-based, CPU). Receives a prompt, fans it out to vllm-alpha
and vllm-beta in parallel, then asks vllm-judge to pick the more accurate
response and explain why.

Runs on CPU because the heavy lifting (generation) happens on the three
GPU endpoints — this worker just coordinates.

The three GPU endpoints (vllm-alpha, vllm-beta, vllm-judge) are Pattern D
image-based endpoints deployed in the same Flash app. Their endpoint IDs
are passed via environment variables so this orchestrator can construct
Endpoint(id=...) clients at runtime. Find the IDs after deploy with:

    flash env get production

Then set them on the orchestrator endpoint (via the RunPod console or
by re-deploying with updated env). For local testing, set them in your
shell before running this script.

Input payload:
    {
        "prompt": "<question for the two models>",
        "max_tokens": 256,            # optional, default 256
        "judge_max_tokens": 512,       # optional, default 512
    }

Output:
    {
        "prompt": "...",
        "alpha": {"text": "...", "elapsed_s": 1.23},
        "beta":  {"text": "...", "elapsed_s": 1.45},
        "judge": {
            "verdict": "a" | "b" | "tie",
            "rationale": "...",
            "elapsed_s": 2.10,
        },
        "total_elapsed_s": 4.78,
    }

Environment variables:
    VLLM_ALPHA_ENDPOINT_ID  — endpoint ID for vllm-alpha
    VLLM_BETA_ENDPOINT_ID   — endpoint ID for vllm-beta
    VLLM_JUDGE_ENDPOINT_ID  — endpoint ID for vllm-judge
"""

import asyncio
import os
import time

from runpod_flash import Endpoint

orchestrator = Endpoint(
    name="judge-orchestrator",
    cpu="cpu3c-2-4",
    workers=(0, 1),
    env={
        # Endpoint IDs for the three GPU endpoints, sourced from the deploy
        # environment so re-deploys always pick up the current IDs instead of
        # stale hardcoded literals. Populate the shell first with `source env.sh`
        # (which reads live values from `flash env get production`).
        "VLLM_ALPHA_ENDPOINT_ID": os.environ.get("VLLM_ALPHA_ENDPOINT_ID", ""),
        "VLLM_BETA_ENDPOINT_ID": os.environ.get("VLLM_BETA_ENDPOINT_ID", ""),
        "VLLM_JUDGE_ENDPOINT_ID": os.environ.get("VLLM_JUDGE_ENDPOINT_ID", ""),
    },
)


JUDGE_SYSTEM = (
    "You are an impartial judge. You will be given a question and two "
    "candidate answers labelled [A] and [B]. Decide which answer is more "
    "accurate, complete, and helpful. Respond with a JSON object that has "
    "exactly two keys: \"verdict\" (one of \"A\", \"B\", or \"tie\") and "
    "\"rationale\" (one or two sentences explaining your choice)."
)


def _build_judge_user_prompt(question: str, ans_a: str, ans_b: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"[A]: {ans_a}\n\n"
        f"[B]: {ans_b}\n\n"
        f"Which answer is better? Reply with JSON only."
    )


def _parse_verdict(raw: str) -> dict:
    """Best-effort extraction of {verdict, rationale} from the judge output.

    verdict is normalized to "a", "b", or "tie".

    Qwen3.6 outputs thinking blocks (`<|...|>thought\\n...<|...|>`)
    before the final answer. We strip those and look for JSON in the
    remaining content.
    """
    import json
    import re

    # Strip Qwen3.6 thinking blocks: <|channel>thought\n ... <|channel|>
    raw = re.sub(r'<\|[^>]*\>thought\n.*?<\|[^>]*\|>', '', raw, flags=re.DOTALL)
    raw = raw.strip()

    # Try direct JSON parse first
    try:
        obj = json.loads(raw)
        if "verdict" in obj and "rationale" in obj:
            v = str(obj["verdict"]).strip().lower()
            if v not in ("a", "b", "tie"):
                # Unexpected verdict value — surface it instead of masking as a tie.
                return {"verdict": "unparseable", "rationale": str(obj["rationale"])}
            return {"verdict": v, "rationale": str(obj["rationale"])}
    except Exception:
        pass

    # Fallback: regex for {"verdict": "A", "rationale": "..."}
    m = re.search(r'"verdict"\s*:\s*"(?P<v>[AB]|tie)"', raw, re.IGNORECASE)
    verdict = m.group("v").lower() if m else "unparseable"
    m2 = re.search(r'"rationale"\s*:\s*"(?P<r>[^"]*)"', raw, re.IGNORECASE)
    rationale = m2.group("r") if m2 else raw[:200]
    return {"verdict": verdict, "rationale": rationale}


def _get_endpoint(env_var: str) -> Endpoint:
    """Construct an Endpoint(id=...) client from an env var."""
    endpoint_id = os.environ.get(env_var)
    if not endpoint_id:
        raise RuntimeError(
            f"{env_var} not set. Deploy the GPU endpoints first, "
            f"then set this env var on the orchestrator endpoint. "
            f"Find IDs with: flash env get production"
        )
    return Endpoint(id=endpoint_id)


async def _call_vllm(ep: Endpoint, messages: list, max_tokens: int) -> dict:
    """Call a worker-vLLM endpoint via the native (non-OpenAI) queue route.

    worker-vllm's JobInput reads `messages` (list) and `sampling_params` (dict)
    from the job input. When `openai_route` is absent it uses the native
    vLLMEngine, which applies the chat template automatically when
    `llm_input` is a list of messages (engine.py:131) and returns
    [{"choices": [{"tokens": ["<full_text>"]}], "usage": {...}}] for
    non-streaming requests (engine.py:174-181).

    runsync may return IN_QUEUE if the worker is cold-starting. We poll
    with job.wait() until the job completes.
    """
    t0 = time.perf_counter()
    job = await ep.runsync(
        {
            "messages": messages,
            "sampling_params": {"temperature": 0.0, "max_tokens": max_tokens},
            "apply_chat_template": True,
        },
        timeout=600.0,
    )
    # runsync returns immediately with IN_QUEUE on cold start — poll until done
    if not job.done:
        await job.wait(timeout=600.0)
    elapsed = time.perf_counter() - t0
    out = job.output
    text = ""
    if isinstance(out, list):
        # worker-vllm yields batches as a list; take the first (non-streaming = 1 element)
        out = out[0] if out else {}
    if isinstance(out, dict):
        if "error" in out:
            raise RuntimeError(f"vLLM error: {out['error']}")
        choices = out.get("choices", [])
        if choices:
            # native format: choices[0]["tokens"] is a list of strings
            tokens = choices[0].get("tokens", [])
            if isinstance(tokens, list):
                text = "".join(tokens)
            else:
                text = str(tokens)
    elif isinstance(out, str):
        text = out
    return {"text": text, "elapsed_s": round(elapsed, 3)}


@orchestrator
async def judge(prompt: str, max_tokens: int = 256, judge_max_tokens: int = 512) -> dict:
    t_start = time.perf_counter()

    # Construct clients at runtime from env vars. This avoids importing
    # the Pattern D Endpoint definitions at module level, which would
    # cause duplicate endpoint registration during flash build.
    alpha_ep = _get_endpoint("VLLM_ALPHA_ENDPOINT_ID")
    beta_ep = _get_endpoint("VLLM_BETA_ENDPOINT_ID")
    judge_ep = _get_endpoint("VLLM_JUDGE_ENDPOINT_ID")

    user_msg = [{"role": "user", "content": prompt}]

    # Fan out to both inference endpoints in parallel
    alpha_task = _call_vllm(alpha_ep, user_msg, max_tokens)
    beta_task = _call_vllm(beta_ep, user_msg, max_tokens)
    alpha_res, beta_res = await asyncio.gather(alpha_task, beta_task, return_exceptions=True)

    if isinstance(alpha_res, Exception):
        alpha_res = {"text": f"[ERROR] {alpha_res}", "elapsed_s": 0.0}
    if isinstance(beta_res, Exception):
        beta_res = {"text": f"[ERROR] {beta_res}", "elapsed_s": 0.0}

    # Call the judge
    judge_msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": _build_judge_user_prompt(
            prompt, alpha_res["text"], beta_res["text"]
        )},
    ]
    try:
        judge_raw = await _call_vllm(judge_ep, judge_msgs, judge_max_tokens)
        parsed = _parse_verdict(judge_raw["text"])
        judge_out = {
            "verdict": parsed["verdict"],
            "rationale": parsed["rationale"],
            "elapsed_s": judge_raw["elapsed_s"],
        }
    except Exception as e:
        judge_out = {
            "verdict": "error",
            "rationale": f"[judge call failed: {e}]",
            "elapsed_s": 0.0,
        }

    return {
        "prompt": prompt,
        "alpha": alpha_res,
        "beta": beta_res,
        "judge": judge_out,
        "total_elapsed_s": round(time.perf_counter() - t_start, 3),
    }


if __name__ == "__main__":
    import sys

    q = os.getenv("PROMPT", "Explain why the sky is blue in two sentences.")
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])

    print(f"Question: {q!r}")
    result = asyncio.run(judge(q))
    print("\n--- Alpha ---")
    print(result["alpha"]["text"])
    print("\n--- Beta ---")
    print(result["beta"]["text"])
    print("\n--- Judge ---")
    print(f"Verdict: {result['judge']['verdict']}")
    print(f"Rationale: {result['judge']['rationale']}")
    print(f"\nTotal elapsed: {result['total_elapsed_s']}s")