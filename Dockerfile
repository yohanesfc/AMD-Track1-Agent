FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm \
 && python -m spacy download en_core_web_md

# Zero-token local sentiment model (FAQ: local inference counts as zero
# tokens). 123MB, over GitHub's file limit, so it's fetched from the HF hub
# at build time instead of living in the repo.
RUN python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
os.makedirs("/app/local_models/sentiment", exist_ok=True)
repo = "Xenova/twitter-roberta-base-sentiment-latest"
for remote, local in [("onnx/model_quantized.onnx", "model.onnx"),
                      ("tokenizer.json", "tokenizer.json"),
                      ("config.json", "config.json")]:
    shutil.copy(hf_hub_download(repo, remote), f"/app/local_models/sentiment/{local}")
PY

COPY classify.py prompts.py model_select.py fireworks_client.py runner.py local_infer.py ./

# Kill switch for the local pre-filter -- flip to "off" and rebuild to get
# a pure-Fireworks pipeline without touching code.
ENV LOCAL_PREFILTER=on

# TOKEN-count-based overrides (the leaderboard metric is total tokens, not
# dollars), tested 2026-07-11 against the real ALLOWED_MODELS family
# (MiniMax/Kimi) with a live Fireworks key -- see .env for the full
# reasoning. kimi-k2p7-code emits fewer tokens than minimax-m3 on every
# category once reasoning_effort="none" is set for the simple ones.
# model_select.py always re-validates these against the real
# ALLOWED_MODELS at runtime and silently falls back to the size heuristic if
# an id isn't present, so this never bypasses the injected allow-list.
# RE-VERIFY once the official launch-day ALLOWED_MODELS is confirmed -- if
# the exact ids differ from these, rebuild with corrected values before the
# July 11 15:00 UTC deadline.
ENV CHEAP_MODEL_OVERRIDE=accounts/fireworks/models/kimi-k2p7-code
ENV STRONG_MODEL_OVERRIDE=accounts/fireworks/models/kimi-k2p7-code
# Different family for second attempts -- same-model retries fail the same
# way when the primary is rate-limited/down. Validated against the real
# ALLOWED_MODELS at runtime like the other overrides.
ENV RETRY_MODEL_OVERRIDE=accounts/fireworks/models/minimax-m3

# No EXPOSE / no server -- this is a batch job: read /input/tasks.json,
# write /output/results.json, exit. FIREWORKS_API_KEY, FIREWORKS_BASE_URL,
# and ALLOWED_MODELS are injected by the harness at evaluation time -- do
# not bake a .env file into the image.
ENTRYPOINT ["python", "runner.py"]
