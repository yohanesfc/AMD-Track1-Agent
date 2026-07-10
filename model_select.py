"""
Picks a "cheap" and "strong" model from ALLOWED_MODELS at runtime.

ALLOWED_MODELS isn't known until launch day, so nothing here can hardcode
a model ID -- everything is inferred from the list the harness injects.

Heuristic: look for a parameter-count token in the model id (e.g. "8b",
"70b", "0p5b", "405b") and rank ascending. Models with no detectable size
are treated as mid-tier. Smallest -> cheap, largest -> strong. If only one
model is available, it's used for every tier (no routing decision left to
make, but the deterministic pre-filter in classify.py/runner.py still
saves tokens on anything answerable without a call... though for this
track every task needs a real answer, so realistically all tasks call
the model -- the savings come entirely from picking the right tier).
"""
import os
import re

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?|\d+p\d+)\s*b\b", re.I)


def _inferred_size(model_id: str) -> float:
    match = _SIZE_RE.search(model_id.replace("_", "-"))
    if not match:
        return 50.0  # unknown -> assume mid-tier
    raw = match.group(1).lower().replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return 50.0


def load_allowed_models() -> list[str]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise RuntimeError(
            "ALLOWED_MODELS is empty or unset. This must be injected by the "
            "harness at runtime -- for local testing, set it in your .env."
        )
    return models


def select_tiers(models: list[str]) -> dict[str, str]:
    """
    Returns {"cheap": model_id, "strong": model_id, "retry": model_id}.

    "retry" is the second-attempt model after a failed call. It should be
    a DIFFERENT model from the primary when possible: if the primary is
    down or rate-limited (likely on launch day, when every submission
    hammers the same favorite), retrying the same model tends to fail the
    same way. RETRY_MODEL_OVERRIDE picks it explicitly; the fallback is
    the largest allowed model that differs from "strong", or "strong"
    itself when only one model exists.

    Checks CHEAP_MODEL_OVERRIDE / STRONG_MODEL_OVERRIDE env vars first --
    useful because the size-in-name heuristic can be flat-out wrong (e.g.
    gpt-oss-120b is the cheapest model on Fireworks despite having the
    biggest parameter count in its name; smaller-sounding names like
    glm-5p2 can cost far more per token; and model families like
    MiniMax/Kimi don't encode a parameter count in the id at all, so the
    heuristic can't rank them meaningfully). Set these once you've checked
    real pricing for whatever ALLOWED_MODELS actually contains. Each
    override is only honored if it's present in ALLOWED_MODELS -- never
    routes to a model outside the list the harness gave you. Overrides are
    independent: you can set just CHEAP_MODEL_OVERRIDE and let the other
    fall back to the heuristic.
    """
    ranked = sorted(models, key=_inferred_size)
    default_cheap = ranked[0]
    default_strong = ranked[-1]

    cheap_override = os.environ.get("CHEAP_MODEL_OVERRIDE", "").strip()
    strong_override = os.environ.get("STRONG_MODEL_OVERRIDE", "").strip()
    retry_override = os.environ.get("RETRY_MODEL_OVERRIDE", "").strip()

    cheap = cheap_override if cheap_override in models else default_cheap
    strong = strong_override if strong_override in models else default_strong

    if retry_override in models:
        retry = retry_override
    else:
        others = [m for m in ranked if m != strong]
        retry = others[-1] if others else strong

    return {"cheap": cheap, "strong": strong, "retry": retry}


def resolve_model(tier: str, tiers: dict[str, str]) -> str:
    return tiers.get(tier, tiers["strong"])
