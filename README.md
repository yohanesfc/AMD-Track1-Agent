# Track 1 — General-Purpose AI Agent (Token-Efficient Routing)

AMD Developer Hackathon ACT II, Track 1 submission.

## The idea

Track 1 scoring is two-stage: an LLM-Judge accuracy gate first, then — only
for submissions that pass — ranking ascending by total tokens. So the
optimization target isn't "which model is smartest", it's **the cheapest
routing that still passes the gate, per task type**.

## Architecture

```
/input/tasks.json
      |
      v
classify.py -----------------> category (zero cost, deterministic regex,
      |                          no API call) — one of the 8 capability
      |                          categories
      v
prompts.py -------------------> category -> tier (cheap/strong), a short
      |                          category-tailored system prompt, a token
      |                          budget, and the reasoning-effort setting
      v
model_select.py --------------> ALLOWED_MODELS (read from env at runtime,
      |                          never hardcoded) -> {"cheap": ..., "strong": ...}
      v
fireworks_client.py ----------> ONE call to FIREWORKS_BASE_URL with the
      |                          chosen model. Retries once (different model
      |                          family) and falls back to the strong model
      |                          as a last resort, so no task_id is ever
      |                          omitted from the output.
      v
/output/results.json
```

The single biggest token lever is `reasoning_effort="none"` on the four
mechanical categories (sentiment, NER, summarization, factual knowledge):
it suppresses hidden chain-of-thought — ~85–90% of completion tokens on
those categories — with no measured accuracy cost. The four categories that
genuinely need multi-step reasoning (math, logic, code debugging, code
generation) keep a full reasoning budget.

A local-inference pre-filter (`local_infer.py`: int8 ONNX sentiment +
dual-spaCy NER consensus) is included but ships **disabled**
(`LOCAL_PREFILTER=off` in the Dockerfile): A/B evaluation showed the API
path with reasoning switched off is both more accurate and already cheap,
so the accuracy risk wasn't worth the near-zero token saving.

## Category → tier mapping

| Category | Tier | Why |
|---|---|---|
| Sentiment classification | cheap | mechanical, low reasoning depth |
| Named entity recognition | cheap | pattern extraction, low reasoning depth |
| Summarization | cheap | mostly compression, not reasoning |
| Factual knowledge | cheap | recall-heavy, not multi-step |
| Math reasoning | strong | multi-step arithmetic error-prone on small models |
| Logical/deductive reasoning | strong | constraint satisfaction needs real reasoning |
| Code debugging | strong | needs to actually understand the bug |
| Code generation | strong | correctness bar is unforgiving |

Both tiers run the same primary model; the tiers differ by
reasoning-effort setting and token budget, with a different model family
held in reserve as the retry path.

## `ALLOWED_MODELS` handling

The container never calls a model outside whatever `ALLOWED_MODELS` the
harness injects at runtime — that check is unconditional in
`model_select.py`. Optional `CHEAP_MODEL_OVERRIDE` / `STRONG_MODEL_OVERRIDE`
/ `RETRY_MODEL_OVERRIDE` env vars express a *preference*, but each is
validated against the real injected list first; anything not present is
ignored and a parameter-count heuristic assigns tiers instead. Either way
the container produces valid, scoreable output.

## Setup & local testing

```bash
cp .env.example .env   # add your own Fireworks key + model IDs (local testing only)
pip install -r requirements.txt
./scripts/run_local.sh
```

This runs `runner.py` against `data/tasks.json` (8 sample tasks, one per
category) and writes `data/results.json`.

Classifier-only check (no API calls, free):
```bash
python -c "
import json
from classify import classify
for t in json.load(open('data/tasks.json')):
    print(t['task_id'], '->', classify(t['prompt']))
"
```

## Build & image

`.github/workflows/docker-build.yml` builds a **linux/amd64** image on every
push and publishes it to GHCR:

```
ghcr.io/yohanesfc/track1-agent:latest
```

Verify the platform:
```bash
docker manifest inspect ghcr.io/yohanesfc/track1-agent:latest
```

## Container contract test

```bash
mkdir -p /tmp/in /tmp/out
cp data/tasks.json /tmp/in/tasks.json

docker run --rm \
  -v /tmp/in:/input -v /tmp/out:/output \
  -e FIREWORKS_API_KEY=your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/llama-v3p1-70b-instruct \
  ghcr.io/yohanesfc/track1-agent:latest

cat /tmp/out/results.json
```

This mirrors exactly what the judging harness does: mounts `/input` and
`/output`, injects the three env vars, and expects the container to exit 0
with a valid `results.json`.

## Compliance & validation

- Reads `/input/tasks.json`, writes `/output/results.json`; **every task_id
  is always present** in the output, even on failure; exits 0 on success.
- `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS` come purely
  from env — no hardcoded key, URL, or model list. No `.env` in the image.
- No caching or hardcoding of answers — every task is computed at runtime
  by a live Fireworks call.
- Stress-tested at judging scale: 1,200 synthetic tasks (150× the sample
  set, all 8 categories) against the live Fireworks API at concurrency 64
  and 200. All 1,200 answered in ~120s (~4.5× under the time budget), zero
  hard failures, zero lost task_ids. Rate-limit 429s under load were fully
  absorbed by the retry + fallback path.
