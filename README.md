# Track 1 — General-Purpose AI Agent (Token-Efficient Routing)

Built to the actual AMD Developer Hackathon ACT II Participant Guide spec.

## How scoring works (from the participant guide)

1. **Accuracy gate**: an LLM-Judge checks every answer against expected intent.
   Fall below the threshold → excluded from the leaderboard entirely, tokens
   don't matter at that point.
2. **Token efficiency**: submissions that pass the gate are ranked ascending
   by total tokens recorded by the judging proxy. Fewer tokens = higher rank.

Critically: **local models/tokens count as zero.** All inference that counts
must go through `FIREWORKS_BASE_URL`. So the actual routing decision in this
track isn't "local vs. remote" — it's:

1. What can be resolved with **zero-cost deterministic logic** (no model call)?
2. For everything else, which is the **cheapest Fireworks model in
   `ALLOWED_MODELS`** that's still likely to pass the accuracy gate for this
   task's category?

That's what this scaffold does.

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

This is a **safety-biased default**: failing the accuracy gate scores zero
regardless of tokens saved, so the harder four categories default to your
strongest available model. Tune this table once you've tested against your
own eval set — if the cheap model holds accuracy on, say, summarization *and*
sentiment reliably in your testing, that's already reflected here; if it
doesn't hold up on your test set, move that category to `strong` in
`prompts.py`.

## `ALLOWED_MODELS` handling

The submitted image never calls a model outside whatever `ALLOWED_MODELS`
the harness injects at runtime — that check is unconditional in
`model_select.py`. Within that constraint, `select_tiers()` decides which
allowed model is "cheap" and which is "strong":

1. **Optional override** — `CHEAP_MODEL_OVERRIDE` / `STRONG_MODEL_OVERRIDE`
   env vars, if set *and* present in the real `ALLOWED_MODELS`, are used
   directly. This exists because parameter count in a model's name doesn't
   reliably predict its price — `gpt-oss-120b` is the cheapest model on
   Fireworks despite the biggest number in its name, while smaller-sounding
   names can cost far more per token. These are **not** baked into the
   Dockerfile by default (see the compliance checklist below for why, and
   for the one deliberate exception).
2. **Fallback heuristic** — if no override matches, `_inferred_size()`
   parses a parameter-count token out of each model ID (`8b`, `70b`, `0p5b`)
   and ranks ascending: smallest → cheap, largest → strong. This is a
   reasonable default, not a guarantee — as above, it can be flat-out wrong
   on real pricing for at least one known model family.

Practical effect: without the override, tier assignment might occasionally
put the wrong model in the wrong tier (real cost, not accuracy) — the
container still produces valid, scoreable output either way. With a
correctly-set override, tiering matches real pricing. See the compliance
checklist for when/how to set the override for your final submission.

## Setup & local testing

```bash
cp .env.example .env
# edit .env: put in your own Fireworks API key + a couple of real model IDs
# you have access to, for local testing only

pip install -r requirements.txt
./scripts/run_local.sh
```

This runs `runner.py` against `data/tasks.json` (8 sample tasks, one per
category) and prints `data/results.json`. Read the classification output
in the log lines to sanity-check the router's decisions.

Quick correctness check on the classifier alone (no API calls, free):
```bash
python -c "
import json
from classify import classify
for t in json.load(open('data/tasks.json')):
    print(t['task_id'], '->', classify(t['prompt']))
"
```

## Build & push (must be linux/amd64) — via GitHub Actions (recommended)

Your OCI server is ARM64, so a plain `docker build` there produces an ARM64
image that the judging VM can't pull. Easiest fix: let GitHub build it for
you on a real amd64 runner. `.github/workflows/docker-build.yml` is already
set up to do this.

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

That push triggers the workflow automatically. Watch it run under the
**Actions** tab of your repo. On success it pushes:

```
ghcr.io/<your-github-username>/track1-agent:latest
```

**One manual step after the first successful run:** GitHub Container
Registry defaults new packages to **private**. Go to your GitHub profile →
**Packages** → `track1-agent` → **Package settings** → change visibility to
**Public**. The submission rules require the image to be publicly pullable —
double check this before submitting.

Verify it's really `linux/amd64`:
```bash
docker manifest inspect ghcr.io/<you>/track1-agent:latest
```

## Build & push manually (only if you're not using GitHub Actions)

The judging VM runs `linux/amd64`. If you're building locally on Apple
Silicon or your ARM64 OCI box:

```bash
docker buildx build --platform linux/amd64 --tag ghcr.io/<you>/track1-agent:latest --push .
```

Otherwise a standard build is fine:
```bash
docker build --tag ghcr.io/<you>/track1-agent:latest .
docker push ghcr.io/<you>/track1-agent:latest
```

## Test the real container contract locally

```bash
mkdir -p /tmp/in /tmp/out
cp data/tasks.json /tmp/in/tasks.json

docker run --rm \
  -v /tmp/in:/input -v /tmp/out:/output \
  -e FIREWORKS_API_KEY=your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/llama-v3p1-70b-instruct \
  ghcr.io/<you>/track1-agent:latest

cat /tmp/out/results.json
```

This mirrors exactly what the harness does: mounts `/input` and `/output`,
injects the three env vars, and expects the container to exit 0 with a
valid `results.json`.

## Compliance checklist (from the participant guide)

- [x] Reads `/input/tasks.json`, writes `/output/results.json`
- [x] Reads `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS`
      purely from env — no hardcoded key or URL. `model_select.py` optionally
      accepts `CHEAP_MODEL_OVERRIDE`/`STRONG_MODEL_OVERRIDE` as a *preference*,
      but always validates them against the real `ALLOWED_MODELS` at runtime
      and falls back to the naming heuristic if either isn't present — it
      never calls a model outside the injected list. The Dockerfile itself
      does not bake these in (see the final-step item below).
- [x] No `.env` bundled in the image (`.dockerignore`)
- [x] Every task_id always present in output, even on failure (empty-string
      fallback rather than omission)
- [x] Exits 0 on success; uncaught top-level errors exit non-zero
- [x] No caching/hardcoding of answers — every task hits a live call (or the
      zero-cost classifier) at runtime
- [ ] **You still need to**: confirm total runtime stays under 10 minutes for
      the real (larger, hidden) task set — tune `MAX_CONCURRENCY` in `.env`
      if needed. Current default is 6 concurrent requests.
- [ ] **You still need to**: push to GitHub (workflow builds + pushes
      `linux/amd64` automatically) — then flip the GHCR package to **public**
      before the July 11, 15:00 UTC deadline.
- [ ] **Do this last, right before submitting**: once the official launch-day
      `ALLOWED_MODELS` is confirmed, verify (or re-verify) which of those
      models is actually cheapest/strongest by real pricing — don't trust
      parameter count in the name (`gpt-oss-120b` is the cheapest model on
      Fireworks despite having the largest number in its name). Add
      `CHEAP_MODEL_OVERRIDE` / `STRONG_MODEL_OVERRIDE` as `ENV` lines in the
      Dockerfile with the confirmed-correct values, then rebuild and push
      one final time. Skipping this isn't a hard failure — the heuristic
      fallback still produces a valid, scoreable submission — but tier
      assignment may end up backwards on cost, wasting tokens on the ranking
      even though accuracy is unaffected.

## Where to spend your remaining time

1. Run `scripts/run_local.sh` against harder/adversarial variants of each
   category (unseen prompt variants are what's actually evaluated) and watch
   for any category where the `cheap` tier visibly struggles — move it to
   `strong` in `prompts.py` if so.
2. Once `ALLOWED_MODELS` is published, re-check `select_tiers()` picked the
   models you'd expect.
3. Build, push, and run the container contract test above end-to-end at
   least once before the deadline — a broken `/input`/`/output` contract or
   a non-`linux/amd64` image scores zero regardless of how good the routing
   logic is.
