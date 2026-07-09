"""
Entrypoint for the Track 1 submission container.

Reads  : $TASKS_INPUT_PATH  (default /input/tasks.json)
Writes : $TASKS_OUTPUT_PATH (default /output/results.json)

Per task:
  1. classify() -- zero-cost, local, no Fireworks call
  2. pick model tier for that category (cheap vs strong)
  3. one Fireworks call with a category-tailored, concise prompt
  4. on failure, retry once; if still failing, fall back to the strong
     model once more before giving up (never omit a task_id from output)

Always writes a valid results.json and exits 0, even if some individual
tasks failed after retries, timed out against the global deadline, or
were malformed in the input -- a missing/malformed *file* scores zero for
everything (unrecoverable, so that case still exits non-zero), but every
task_id that was actually present in the input is guaranteed a row in the
output, empty-string answer at worst.
"""
import asyncio
import json
import os
import sys
import time

from classify import classify
from prompts import SYSTEM_PROMPTS, TIER_BY_CATEGORY, MAX_TOKENS_BY_CATEGORY
from model_select import load_allowed_models, select_tiers, resolve_model
from fireworks_client import FireworksClient

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("TASKS_OUTPUT_PATH", "/output/results.json")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "6"))
MAX_RUNTIME_SECONDS = 9 * 60  # stay safely under the 10-minute hard limit


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


SLOW_CALL_WARNING_MS = 20000  # flag anything within striking distance of the 30s limit


async def solve_task(client: FireworksClient, tiers: dict, sem: asyncio.Semaphore, task: dict) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]
    category = classify(prompt)
    tier = TIER_BY_CATEGORY.get(category, "strong")
    model = resolve_model(tier, tiers)
    system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["factual_knowledge"])
    max_tokens = MAX_TOKENS_BY_CATEGORY.get(category, 300)

    async with sem:
        for attempt, use_model in enumerate([model, tiers["strong"]]):
            try:
                answer, tokens, latency_ms = await client.complete(use_model, system_prompt, prompt, max_tokens)
                if answer:
                    warn = " ⚠ SLOW (near 30s limit)" if latency_ms >= SLOW_CALL_WARNING_MS else ""
                    log(
                        f"[ok] {task_id} category={category} tier={tier} model={use_model} "
                        f"tokens={tokens} latency={latency_ms}ms{warn}"
                    )
                    return {"task_id": task_id, "answer": answer}
            except Exception as exc:  # noqa: BLE001
                log(f"[retry] {task_id} attempt={attempt} model={use_model} error={exc}")
                await asyncio.sleep(1)
        # Both attempts failed -- never omit the task_id.
        log(f"[fail] {task_id} giving up after retries")
        return {"task_id": task_id, "answer": ""}


def _load_tasks(path: str) -> list[dict]:
    """Keeps any task with a task_id even if other fields are malformed, so
    every task_id in the input is still guaranteed a row in the output --
    only a task missing task_id entirely (nothing to key the result on) is
    dropped."""
    with open(path) as f:
        raw_tasks = json.load(f)

    tasks = []
    for i, t in enumerate(raw_tasks):
        task_id = t.get("task_id") if isinstance(t, dict) else None
        if not task_id:
            log(f"[skip] malformed task at index {i}, no task_id: {t!r}")
            continue
        tasks.append({"task_id": task_id, "prompt": t.get("prompt") or ""})

    if len(tasks) != len(raw_tasks):
        log(f"Loaded {len(tasks)} usable tasks from {path} ({len(raw_tasks) - len(tasks)} skipped)")
    else:
        log(f"Loaded {len(tasks)} tasks from {path}")
    return tasks


async def run():
    start = time.monotonic()

    tasks = _load_tasks(INPUT_PATH)

    models = load_allowed_models()
    tiers = select_tiers(models)
    log(f"Model tiers -> cheap: {tiers['cheap']} | strong: {tiers['strong']}")

    client = FireworksClient()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    results = []
    if tasks:
        running = [asyncio.ensure_future(solve_task(client, tiers, sem, t)) for t in tasks]
        _done, pending = await asyncio.wait(running, timeout=MAX_RUNTIME_SECONDS)

        for coro, t in zip(running, tasks):
            if coro in pending:
                coro.cancel()
                log(f"[timeout] {t['task_id']} still running at the {MAX_RUNTIME_SECONDS}s deadline, marking failed")
                results.append({"task_id": t["task_id"], "answer": ""})
            else:
                results.append(coro.result())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.monotonic() - start
    log(f"Wrote {len(results)} results to {OUTPUT_PATH} in {elapsed:.1f}s")


def main():
    try:
        asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL: {exc}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
