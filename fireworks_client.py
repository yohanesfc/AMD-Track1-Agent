"""
All Fireworks calls must go through FIREWORKS_BASE_URL with FIREWORKS_API_KEY
-- both read from the environment, never hardcoded (per submission rules;
calls that bypass this URL aren't recorded and score zero tokens, which
would also mean zero accuracy credit).
"""
import os
import httpx

REQUEST_TIMEOUT = 25.0  # keep headroom under the 30s per-request rule


class FireworksClient:
    def __init__(self):
        self.base_url = os.environ["FIREWORKS_BASE_URL"].rstrip("/")
        self.api_key = os.environ["FIREWORKS_API_KEY"]

    async def complete(
        self, model: str, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> tuple[str, int]:
        """Returns (answer_text, tokens_used)."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()

        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            # Reasoning models (e.g. gpt-oss) can omit "content" entirely if
            # they exhaust max_tokens on hidden reasoning before writing a
            # final answer. Fall back to reasoning_content if present, purely
            # so we have *something* to work with -- but hitting this branch
            # is a signal max_tokens is too tight for this model/category.
            content = (message.get("reasoning_content") or "").strip()
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return content, tokens
