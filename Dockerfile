FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY classify.py prompts.py model_select.py fireworks_client.py runner.py ./

# Pricing-based overrides, tested 2026-07-09 against the real ALLOWED_MODELS
# family (MiniMax/Kimi/Gemma) with a live Fireworks key -- see .env for the
# full reasoning. model_select.py always re-validates these against the real
# ALLOWED_MODELS at runtime and silently falls back to the size heuristic if
# either id isn't present, so this never bypasses the injected allow-list.
# RE-VERIFY once the official launch-day ALLOWED_MODELS is confirmed -- if
# the exact ids differ from these, rebuild with corrected values before the
# July 11 15:00 UTC deadline.
ENV CHEAP_MODEL_OVERRIDE=accounts/fireworks/models/minimax-m3
ENV STRONG_MODEL_OVERRIDE=accounts/fireworks/models/kimi-k2p7-code
ENV STRONG_CODE_MODEL_OVERRIDE=accounts/fireworks/models/kimi-k2p7-code

# No EXPOSE / no server -- this is a batch job: read /input/tasks.json,
# write /output/results.json, exit. FIREWORKS_API_KEY, FIREWORKS_BASE_URL,
# and ALLOWED_MODELS are injected by the harness at evaluation time -- do
# not bake a .env file into the image.
ENTRYPOINT ["python", "runner.py"]
