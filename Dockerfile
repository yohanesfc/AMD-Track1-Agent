FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY classify.py prompts.py model_select.py fireworks_client.py runner.py ./

# No EXPOSE / no server -- this is a batch job: read /input/tasks.json,
# write /output/results.json, exit. Env vars (FIREWORKS_API_KEY,
# FIREWORKS_BASE_URL, ALLOWED_MODELS) are injected by the harness at
# evaluation time -- do not bake a .env file into the image.
ENTRYPOINT ["python", "runner.py"]
