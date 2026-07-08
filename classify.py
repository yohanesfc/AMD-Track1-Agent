"""
Deterministic, zero-token category classifier.

This runs entirely locally -- no Fireworks call, no cost -- and decides
which of the 8 capability categories a prompt most likely belongs to.
The category then drives both the system prompt and the model tier
(see model_select.py), which is where the actual token savings come from:
we only pay for exactly the model capability each task needs.

Heuristic, not perfect -- that's fine. Misclassification mostly costs a
slightly-too-strong (safe) or slightly-too-weak (risky) model choice, not
a hard failure. Biased toward the safer direction (escalate on ambiguity).
"""
import re

CATEGORIES = (
    "code_debugging",
    "code_generation",
    "logical_reasoning",
    "math_reasoning",
    "named_entity_recognition",
    "sentiment_classification",
    "summarization",
    "factual_knowledge",  # fallback / default
)

_CODE_MARKERS = re.compile(
    r"```|\bdef \w+\(|\bfunction\s+\w+\(|\bclass\s+\w+[:\(]|;\s*$|=>|\bimport\s+\w+", re.MULTILINE
)
_BUG_WORDS = re.compile(
    r"\b(bug|fix|error|broken|not working|traceback|exception|incorrect output|debug)\b", re.I
)
_GEN_WORDS = re.compile(
    r"\b(write (a |an )?(\w+\s+){0,2}function|implement (a |an )?function|write code|"
    r"create (a |an )?function|write a program)\b", re.I
)
_MATH_KEYWORDS = re.compile(
    r"\b(calculate|percent|percentage|total|sum|how many|how much|average|ratio|"
    r"profit|discount|interest rate|projection|depreciat)\b",
    re.I,
)
_MATH_SYMBOLS = re.compile(r"\$\d|\d+\s*%")
_MATH_NUMS = re.compile(r"\d+(\.\d+)?")
_LOGIC_WORDS = re.compile(
    r"\b(either .* or|neither .* nor|if and only if|exactly one of|all of the following conditions|"
    r"who is the|which one of|logic puzzle|satisfies (all|every) (the )?condition|cannot both)\b", re.I
)
_NER_WORDS = re.compile(
    r"\b(extract (the )?(named )?entit(y|ies)|named entit(y|ies)|"
    r"identify (all )?(people|persons|organizations|locations|dates)|"
    r"list the (people|organizations|locations|dates))\b", re.I
)
_SENTIMENT_WORDS = re.compile(
    r"\b(sentiment|positive, negative|positive or negative|how does .* feel|"
    r"classify the (tone|emotion))\b", re.I
)
_SUMMARY_WORDS = re.compile(
    r"\b(summari[sz]e|summary|condense|tl;dr|in one sentence|in \d+ words|shorten)\b", re.I
)


def classify(prompt: str) -> str:
    text = prompt.strip()

    if _BUG_WORDS.search(text) and _CODE_MARKERS.search(text):
        return "code_debugging"
    if _GEN_WORDS.search(text):
        return "code_generation"
    if _CODE_MARKERS.search(text) and _BUG_WORDS.search(text):
        return "code_debugging"

    if _LOGIC_WORDS.search(text):
        return "logical_reasoning"

    if (_MATH_KEYWORDS.search(text) or _MATH_SYMBOLS.search(text)) and len(_MATH_NUMS.findall(text)) >= 1:
        return "math_reasoning"

    if _NER_WORDS.search(text):
        return "named_entity_recognition"

    if _SENTIMENT_WORDS.search(text):
        return "sentiment_classification"

    if _SUMMARY_WORDS.search(text):
        return "summarization"

    return "factual_knowledge"
