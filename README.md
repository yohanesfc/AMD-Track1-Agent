# Track 1 — General-Purpose AI Agent (Token-Efficient Routing)

Built to the AMD Developer Hackathon ACT II Participant Guide spec.

## Overview

This agent processes a batch of tasks and routes each one to the most
cost-effective Fireworks model capable of answering it correctly. It
optimizes for the competition's two-stage scoring model: pass an accuracy
gate first, then minimize total tokens among submissions that pass.

## How Scoring Works

1. **Accuracy gate**: an LLM-Judge checks every answer against expected
   intent. Falling below the threshold excludes the submission from the
   leaderboard entirely — token count becomes irrelevant at that point.
2. **Token efficiency**: submissions that pass the gate are ranked
   ascending by total tokens recorded by the judging proxy. Fewer tokens
   ranks higher.

Local models and tokens count as zero. All inference that counts must go
through `FIREWORKS_BASE_URL`. The routing decision in this track is
therefore not "local vs. remote" — it's:

1. What can be resolved with **zero-cost deterministic logic** (no model
   call)?
2. For everything else, which is the **cheapest Fireworks model in
   `ALLOWED_MODELS`** still likely to pass the accuracy gate for that
   task's category?

## Architecture

```
/input/tasks.json
      |
      v
classify.py -----------------> category (zero cost, no Fireworks call)
      |                          one of the 8 capability categories
      v
prompts.py -------------------> category -> tier (cheap/strong) + a short,
                                 category-tailored system prompt + token budget
      |
      v
model_select.py --------------> ALLOWED_MODELS (read from env, never
                                 hardcoded) -> ranked by inferred parameter
                                 count -> {"cheap": ..., "strong": ...}
      |
      v
fireworks_client.py ----------> ONE call to FIREWORKS_BASE_URL with the
                                 chosen model. Retries once on failure,
                                 falls back to the strong model as a
                                 last resort so no task_id is ever omitted.
      |
      v
/output/results.json
```

## Category → Tier Mapping

| Category | Tier | Rationale |
|---|---|---|
| Sentiment classification | cheap | mechanical, low reasoning depth |
| Named entity recognition | cheap | pattern extraction, low reasoning depth |
| Summarization | cheap | mostly compression, not reasoning |
| Factual knowledge | cheap | recall-heavy, not multi-step |
| Math reasoning | strong | multi-step arithmetic error-prone on small models |
| Logical/deductive reasoning | strong | constraint satisfaction needs real reasoning |
| Code debugging | strong | needs to actually understand the bug |
| Code generation | strong | correctness bar is unforgiving |

This is a safety-biased default: failing the accuracy gate scores zero
regardless of tokens saved, so the harder four categories default to the
strongest available model. Tune this table against your own eval set —
move a category to `strong` in `prompts.py` if the cheap tier doesn't
hold up under testing.

## `ALLOWED_MODELS` Handling

`ALLOWED_MODELS` isn't published until launch day, so no model ID may be
hardcoded (the guide explicitly forbids this — calls to un-listed models
invalidate the submission). `model_select.py` parses a parameter-count
token out of each model ID (`8b`, `70b`, `0p5b`, etc.), ranks ascending,
and picks the smallest as `cheap` / largest as `strong`. Unknown-size
models fall back to list order.

### Manual tier override

The size-in-name heuristic can be wrong — a model with a larger parameter
count in its name is not always the more expensive one on Fireworks'
pricing. `model_select.py` supports `CHEAP_MODEL_OVERRIDE` /
`STRONG_MODEL_OVERRIDE` environment variables to pin specific models once
their relative cost has been verified. An override is only honored if it
is present in the `ALLOWED_MODELS` list actually injected at runtime —
otherwise it silently falls back to the naming heuristic. Defaults for
these variables are currently baked into the `Dockerfile` based on
development-time testing; **re-verify them against the official
`ALLOWED_MODELS` list once published on launch day** and rebuild if the
values no longer apply.

## Setup & Local Testing

```bash
cp .env.example .env
# populate .env: Fireworks API key + a couple of real model IDs
# you have access to, for local testing only

pip install -r requirements.txt
./scripts/run_local.sh
```

This runs `runner.py` against `data/tasks.json` (8 sample tasks, one per
category) and writes `data/results.json`. Review the classification
output in the log lines to sanity-check the router's decisions.

Classifier-only correctness check (no API calls, no cost):
```bash
python -c "
import json
from classify import classify
for t in json.load(open('data/tasks.json')):
    print(t['task_id'], '->', classify(t['prompt']))
"
```

## Build & Push (must be linux/amd64) — via GitHub Actions (recommended)

The build host is ARM64, so a local `docker build` produces an ARM64
image the judging VM cannot pull. `.github/workflows/docker-build.yml` is
configured to build on a native amd64 GitHub Actions runner instead.

```bash
cd track1-agent
git init
git add .
git commit -m "Track 1 agent"
gh repo create track1-agent --public --source=. --push
# or manually: create the repo on github.com, then
#   git remote add origin https://github.com/<you>/track1-agent.git
#   git branch -M main
#   git push -u origin main
```

The push triggers the workflow automatically. Monitor it under the
**Actions** tab. On success it publishes:

```
ghcr.io/<your-github-username>/track1-agent:latest
```

**Required manual step after the first successful run:** GitHub
Container Registry defaults new packages to **private**. Navigate to your
GitHub profile → **Packages** → `track1-agent` → **Package settings** →
change visibility to **Public**. The submission rules require the image
to be publicly pullable — verify this before submitting.

Verify the image architecture:
```bash
docker manifest inspect ghcr.io/<you>/track1-agent:latest
```

## Build & Push Manually (only if not using GitHub Actions)

The judging VM runs `linux/amd64`. If building locally on Apple Silicon
or an ARM64 host:

```bash
docker buildx build --platform linux/amd64 --tag ghcr.io/<you>/track1-agent:latest --push .
```

Otherwise a standard build is sufficient:
```bash
docker build --tag ghcr.io/<you>/track1-agent:latest .
docker push ghcr.io/<you>/track1-agent:latest
```

## Testing the Container Contract Locally

```bash
mkdir -p /tmp/in /tmp/out
cp data/tasks.json /tmp/in/tasks.json

source .env

docker run --rm --platform linux/amd64 \
  -v /tmp/in:/input -v /tmp/out:/output \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  ghcr.io/<you>/track1-agent:latest

cat /tmp/out/results.json
```

This mirrors the harness contract exactly: mounts `/input` and
`/output`, injects the three env vars, and expects exit 0 with a valid
`results.json`.

**Note on ARM64 test hosts:** running an amd64 image on an ARM64 machine
without emulation fails with `exec format error`. Register QEMU emulation
first if testing locally on ARM64:
```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```
The judging VM runs natively on `linux/amd64`, so this only affects local
testing — a correct manifest and a passing `docker manifest inspect` are
the source of truth for architecture compliance, not a local emulated
run.

## Compliance Checklist

- [x] Reads `/input/tasks.json`, writes `/output/results.json`
- [x] Reads `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS`
      purely from env — no hardcoded key, URL, or model ID in application
      logic
- [x] No `.env` bundled in the image (`.dockerignore`)
- [x] Every `task_id` always present in output, even on failure (empty
      string fallback rather than omission)
- [x] Exits 0 on success; uncaught top-level errors exit non-zero
- [x] No caching/hardcoding of answers — every task hits a live call (or
      the zero-cost classifier) at runtime
- [ ] **Outstanding**: confirm total runtime stays under 10 minutes for
      the real (larger, hidden) task set — tune `MAX_CONCURRENCY` in
      `.env` if needed. Current default is 6 concurrent requests.
- [ ] **Outstanding**: push to GitHub (workflow builds + pushes
      `linux/amd64` automatically) — then flip the GHCR package to
      **public** before the July 11, 15:00 UTC deadline.

## Known Trade-off: Baked-in Tier Overrides

`Dockerfile` sets `CHEAP_MODEL_OVERRIDE` / `STRONG_MODEL_OVERRIDE` as
image defaults. This is a deliberate trade-off: the harness only injects
`FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`, so
without baking these in, every scored run would fall back to the naming
heuristic — which is known to misprice at least one tested model. The
override is validated against the runtime `ALLOWED_MODELS` list before
use and no-ops safely if the official launch-day list differs from what
was available during development. Re-verify and rebuild once the official
list is confirmed.

## Where to Spend Remaining Time

1. Run `scripts/run_local.sh` against harder/adversarial variants of each
   category (unseen prompt variants are what's actually evaluated) and
   watch for any category where the `cheap` tier visibly struggles — move
   it to `strong` in `prompts.py` if so.
2. Once `ALLOWED_MODELS` is published, re-check `select_tiers()` picked
   the models expected, and confirm the baked-in overrides still apply.
3. Build, push, and run the container contract test above end-to-end at
   least once before the deadline — a broken `/input`/`/output` contract
   or a non-`linux/amd64` image scores zero regardless of routing logic
   quality.
