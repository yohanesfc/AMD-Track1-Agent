FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY classify.py prompts.py model_select.py fireworks_client.py runner.py ./

# Baked-in tier preference, verified against real Fireworks pricing (see
# README). These are NOT hardcoded model calls -- model_select.py still
# validates both are present in the harness's real ALLOWED_MODELS at
# runtime before using them, and falls back to the naming heuristic if
# either is missing (e.g. if the official launch-day list differs from
# what was available during development). This exists because the harness
# only injects FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS --
# it has no reason to know about these custom override vars, so without
# baking them in here, every scored run would fall back to the heuristic,
# which is demonstrably wrong for at least one real model (gpt-oss-120b
# is the cheapest model on Fireworks despite the biggest parameter count
# in its name). Re-verify these once the official launch-day list is
# confirmed and rebuild if it differs from what's baked in below.
ENV CHEAP_MODEL_OVERRIDE="accounts/fireworks/models/gpt-oss-120b"
ENV STRONG_MODEL_OVERRIDE="accounts/fireworks/models/glm-5p2"

# No EXPOSE / no server -- this is a batch job: read /input/tasks.json,
# write /output/results.json, exit. FIREWORKS_API_KEY, FIREWORKS_BASE_URL,
# and ALLOWED_MODELS are injected by the harness at evaluation time -- do
# not bake a .env file into the image.
ENTRYPOINT ["python", "runner.py"]
