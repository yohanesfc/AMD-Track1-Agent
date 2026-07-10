"""
Zero-token local pre-filter for sentiment_classification and
named_entity_recognition.

The FAQ is explicit that local inference counts as ZERO tokens and is
encouraged ("run as many local models as you need ... so you need to make
as few external API calls"). Accuracy is still judged, so this layer only
answers when it is very sure and escalates everything else to the normal
Fireworks path (~130 tokens/task after reasoning_effort tuning). The cost
asymmetry drives every threshold below: a wrong escalation wastes ~130
tokens, a wrong local answer risks the accuracy gate -- so precision is
prioritized over coverage, and any doubt returns None.

Kill switch: LOCAL_PREFILTER=off disables the whole layer (compliance or
emergency rollback without a code change). try_local() never raises: any
internal failure means None (escalate), never a broken task.

Sentiment: RoBERTa 3-class ONNX (int8), softmax gate:
  - positive/negative accepted at p >= SENTIMENT_GATE (default 0.90)
  - "neutral" predictions ALWAYS escalate -- mixed reviews are where the
    small model and an LLM judge most often disagree
  - contrast markers ("but", "however"...) escalate: mixed-sentiment text
    is exactly the ambiguity the gate exists to catch
NER: consensus of spaCy en_core_web_sm AND en_core_web_md. Tested
2026-07-11: each model alone mislabels confidently in different ways on
the same sentence (sm: "Fireworks AI"->PERSON, "DeepMind"->PRODUCT; md:
"DeepMind"->PERSON) and spaCy exposes no confidence score to gate on --
so agreement between two independently-trained models IS the confidence
signal. Both must produce the identical grouped entity list, every entity
must fall inside the four output groups, and a recall proxy (uncovered
capitalized spans / date-like tokens) must come up empty; any disagreement
or leftover escalates.
"""
import os
import re

_SENTIMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_models", "sentiment")
_SENTIMENT_LABELS = ("negative", "neutral", "positive")  # id2label order in config.json

_sentiment_session = None
_sentiment_tokenizer = None
_nlp_models = None


def _enabled() -> bool:
    return os.environ.get("LOCAL_PREFILTER", "on").strip().lower() not in ("off", "0", "false")


def _sentiment_gate() -> float:
    try:
        return float(os.environ.get("SENTIMENT_GATE", "0.90"))
    except ValueError:
        return 0.90


def _load_sentiment():
    global _sentiment_session, _sentiment_tokenizer
    if _sentiment_session is None:
        import onnxruntime
        from tokenizers import Tokenizer
        _sentiment_session = onnxruntime.InferenceSession(
            os.path.join(_SENTIMENT_DIR, "model.onnx"), providers=["CPUExecutionProvider"]
        )
        _sentiment_tokenizer = Tokenizer.from_file(os.path.join(_SENTIMENT_DIR, "tokenizer.json"))
        _sentiment_tokenizer.enable_truncation(max_length=512)
    return _sentiment_session, _sentiment_tokenizer


def _load_nlp():
    global _nlp_models
    if _nlp_models is None:
        import spacy
        _nlp_models = (spacy.load("en_core_web_sm"), spacy.load("en_core_web_md"))
    return _nlp_models


_QUOTED = re.compile(r"['\"“‘](.{10,4000}?)['\"”’]", re.S)


def _extract_passage(prompt: str) -> str | None:
    """Isolate the text-to-analyze from the instruction around it. The
    instruction half must not leak into the model input (its words would
    skew sentiment and add fake entities), so if the passage can't be
    isolated cleanly, the whole task escalates."""
    quoted = _QUOTED.findall(prompt)
    if quoted:
        return max(quoted, key=len).strip()
    head, sep, tail = prompt.partition(":")
    if sep and len(tail.strip()) >= 10:
        return tail.strip()
    return None


# Mixed-sentiment signals: the exact case where a strict lexical read and
# an LLM judge diverge (e.g. "late and damaged, BUT support fixed it").
_CONTRAST = re.compile(r"\b(but|however|although|though|except|aside from|on the other hand)\b", re.I)


def _local_sentiment(prompt: str) -> str | None:
    text = _extract_passage(prompt)
    if not text:
        return None
    if _CONTRAST.search(text):
        return None

    import numpy as np
    session, tokenizer = _load_sentiment()
    enc = tokenizer.encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.ones_like(ids)
    (logits,) = session.run(None, {"input_ids": ids, "attention_mask": mask})
    exp = np.exp(logits[0] - logits[0].max())
    probs = exp / exp.sum()
    idx = int(probs.argmax())
    label = _SENTIMENT_LABELS[idx]

    if label == "neutral" or float(probs[idx]) < _sentiment_gate():
        return None
    return (
        f"Sentiment: {label}. Reason: the text expresses a clearly {label} "
        "overall tone with no significant opposing signals."
    )


_ENTITY_GROUPS = (
    ("Person", ("PERSON",)),
    ("Organization", ("ORG",)),
    ("Location", ("GPE", "LOC", "FAC")),
    ("Date", ("DATE",)),
)
_GROUP_BY_LABEL = {label: group for group, labels in _ENTITY_GROUPS for label in labels}
_CAP_SPAN = re.compile(r"(?:[A-Z][\w'’-]*)(?:\s+[A-Z][\w'’-]*)*")
_DATE_LIKE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December|\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
)
_RELATIVE_DATES = {"yesterday", "today", "tomorrow", "tonight", "now"}


def _grouped_entities(doc) -> dict[str, list[str]] | None:
    """Maps one spaCy doc to {group: [entity texts]}. Returns None (meaning
    escalate) if any entity falls outside the four output groups -- an
    out-of-group label like PRODUCT is usually this model misreading a
    Person/Org, and silently dropping it would hand the judge an
    incomplete list."""
    grouped: dict[str, list[str]] = {}
    for ent in doc.ents:
        group = _GROUP_BY_LABEL.get(ent.label_)
        if group is None:
            if ent.label_ in ("CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "TIME"):
                continue  # numeric/time chatter, never one of the four groups
            return None
        values = grouped.setdefault(group, [])
        if ent.text not in values:
            values.append(ent.text)
    return grouped


def _local_ner(prompt: str) -> str | None:
    text = _extract_passage(prompt)
    if not text:
        return None

    nlp_sm, nlp_md = _load_nlp()
    doc_sm, doc_md = nlp_sm(text), nlp_md(text)
    if not doc_sm.ents or not doc_md.ents:
        return None

    # Consensus gate: both models must produce the identical grouped list.
    grouped = _grouped_entities(doc_sm)
    if grouped is None or not grouped or grouped != _grouped_entities(doc_md):
        return None

    # A lone relative-date "entity" ("yesterday") on otherwise entity-free
    # text is technically a DATE but almost never what an extraction task
    # is after -- escalate and let the LLM decide.
    flat = [v for values in grouped.values() for v in values]
    if all(v.lower() in _RELATIVE_DATES for v in flat):
        return None

    # Recall proxy: an uncovered capitalized span (except a lone
    # sentence-opening word, which is usually just capitalization) or an
    # uncovered date-like token means spaCy may have missed an entity ->
    # escalate rather than hand the judge an incomplete list.
    covered = [(e.start_char, e.end_char) for e in list(doc_sm.ents) + list(doc_md.ents)]

    def _is_covered(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in covered)

    sentence_starts = {s.start_char for s in doc_sm.sents}
    for m in _CAP_SPAN.finditer(text):
        if _is_covered(m.start(), m.end()):
            continue
        if m.start() in sentence_starts and " " not in m.group():
            continue
        return None
    for m in _DATE_LIKE.finditer(text):
        if not _is_covered(m.start(), m.end()):
            return None

    lines = [
        f"{group_name}: {', '.join(grouped[group_name])}"
        for group_name, _ in _ENTITY_GROUPS
        if grouped.get(group_name)
    ]
    return "\n".join(lines) if lines else None


def try_local(category: str, prompt: str) -> str | None:
    """Returns a fully formatted answer, or None to escalate to Fireworks.
    Never raises."""
    if not _enabled():
        return None
    try:
        if category == "sentiment_classification":
            return _local_sentiment(prompt)
        if category == "named_entity_recognition":
            return _local_ner(prompt)
    except Exception:
        return None
    return None
