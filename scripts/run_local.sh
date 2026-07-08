#!/usr/bin/env bash
# Local dev helper: loads .env, runs runner.py against data/tasks.json,
# prints results.json. This is NOT what runs in the container -- the
# container reads env vars injected by the harness directly (no .env).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "No .env found -- copy .env.example to .env and fill in your Fireworks key first."
  exit 1
fi

set -a
source .env
set +a

python3 runner.py

echo
echo "--- results.json ---"
cat "${TASKS_OUTPUT_PATH:-data/results.json}"
