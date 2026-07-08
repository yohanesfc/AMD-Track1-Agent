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
    # Routed to a coding-specialized strong model rather than the general
    # "strong" tier -- see model_select.py's "strong_code" tier.
    "code_debugging": "strong_code",
    "code_generation": "strong_code",
}

SYSTEM_PROMPTS = {
    "factual_knowledge": (
        "Your response MUST begin with the answer itself as the first "
        "sentence -- no numbered analysis, no restating the question, no "
        "preamble. 2-4 sentences total."
    ),
    "math_reasoning": (
        "Your response MUST begin with 'Answer: <value>' as the very first "
        "line, before anything else. After that line only, you may add one "
        "short line of essential working. Do not narrate a numbered "
        "analysis process or restate the question."
    ),
    "sentiment_classification": (
        "Your response MUST begin with the exact line "
        "'Sentiment: <positive|negative|neutral>. Reason: <short clause>.' "
        "as the very first line, with nothing before it. No numbered "
        "analysis steps, no restating the input text, no second-guessing "
        "out loud."
    ),
    "summarization": (
        "Your response MUST begin with the summary itself as the very "
        "first word -- no numbered analysis, no draft/revision process, no "
        "restating the instructions. Strictly follow any length/format "
        "constraint stated in the prompt."
    ),
    "named_entity_recognition": (
        "Your response MUST begin with the entity list itself as the very "
        "first line -- no numbered analysis steps, no preamble. Group by "
        "type (Person/Organization/Location/Date), one line per type, "
        "comma-separated values, omit empty types."
    ),
    "code_debugging": (
        "Your response MUST begin with the code block itself as the very "
        "first thing -- no numbered analysis steps before it. Return the "
        "corrected code in a single code block, with a one-line comment "
        "above each fix. Nothing outside the code block."
    ),
    "logical_reasoning": (
        "Your response MUST begin with 'Answer: <conclusion>' as the very "
        "first line, before any reasoning. After that line only, justify "
        "it in 1-2 sentences maximum. Do not show a numbered step-by-step "
        "deduction process before the answer line -- reason internally, "
        "but the answer line comes first no matter what."
    ),
    "code_generation": (
        "Your response MUST begin with the code block itself as the very "
        "first thing -- no numbered analysis steps before it. Return the "
        "requested code in a single code block with a one-line docstring. "
        "Nothing outside the code block."
    ),
}

# Rough output budget per category -- generous enough to survive a model
# that leaks some reasoning before the final answer despite instructions
# (observed with both glm-5p1 and glm-5p2), and to give reasoning models
# (gpt-oss-120b et al.) enough room to finish their hidden reasoning *and*
# still write a final answer instead of getting cut off mid-thought.
MAX_TOKENS_BY_CATEGORY = {
    "factual_knowledge": 800,
    "math_reasoning": 900,
    "sentiment_classification": 600,
    "summarization": 700,
    "named_entity_recognition": 900,
    "code_debugging": 1200,
    "logical_reasoning": 1500,
    "code_generation": 1200,
}
