"""
Per-category system prompts (kept short -- every prompt token counts) and
the tier each category is routed to by default.

Tiering logic: categories where a small model is reliably accurate go to
"cheap"; categories with more room for reasoning failure go to "strong",
since failing the accuracy gate scores zero regardless of tokens saved.
"""

TIER_BY_CATEGORY = {
    "sentiment_classification": "cheap",
    "named_entity_recognition": "cheap",
    "summarization": "cheap",
    "factual_knowledge": "cheap",
    "math_reasoning": "strong",
    "logical_reasoning": "strong",
    "code_debugging": "strong",
    "code_generation": "strong",
}

SYSTEM_PROMPTS = {
    "factual_knowledge": (
        "Answer in 2-4 sentences, directly and accurately. Do not show a "
        "numbered analysis process, do not restate the question, do not "
        "explain your approach. Output only the final answer."
    ),
    "math_reasoning": (
        "Your response MUST begin with 'Answer: <value>' as the very first "
        "line, before anything else. After that line only, you may add one "
        "short line of essential working. Do not narrate a numbered "
        "analysis process or restate the question."
    ),
    "sentiment_classification": (
        "Output only one line, exactly in this format: "
        "'Sentiment: <positive|negative|neutral>. Reason: <short clause>.' "
        "No numbered analysis steps, no restating the input text."
    ),
    "summarization": (
        "Output only the summary itself, strictly following any length or "
        "format constraint stated in the prompt. No numbered analysis "
        "steps, no preamble, no restating the instructions."
    ),
    "named_entity_recognition": (
        "Output only the entities, grouped by type (Person/Organization/"
        "Location/Date), one line per type, comma-separated values. Omit "
        "empty types. No numbered analysis steps, no preamble."
    ),
    "code_debugging": (
        "Output only the corrected code in a single code block, with a "
        "one-line comment above each fix explaining it. No numbered "
        "analysis steps, no explanation outside the code block."
    ),
    "logical_reasoning": (
        "Your response MUST begin with 'Answer: <conclusion>' as the very "
        "first line, before any reasoning. After that line only, justify "
        "it in 1-2 sentences maximum. Do not show a numbered step-by-step "
        "deduction process before the answer line -- reason internally, "
        "but the answer line comes first no matter what."
    ),
    "code_generation": (
        "Output only the requested code in a single code block with a "
        "one-line docstring. No numbered analysis steps, no explanation "
        "outside the code block."
    ),
}

# Rough output budget per category -- generous enough to survive a model
# that leaks some reasoning before the final answer despite instructions
# (observed with glm-5p1), and to give reasoning models (gpt-oss-120b et
# al.) enough room to finish their hidden reasoning *and* still write a
# final answer instead of getting cut off mid-thought.
MAX_TOKENS_BY_CATEGORY = {
    "factual_knowledge": 650,
    "math_reasoning": 900,
    "sentiment_classification": 250,
    "summarization": 350,
    "named_entity_recognition": 300,
    "code_debugging": 1200,
    "logical_reasoning": 1500,
    "code_generation": 1200,
}
