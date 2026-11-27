"""Evaluation metrics (no heavy dependencies)."""
import re
import string
from collections import Counter
from typing import Optional

_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation and articles."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains_gold(prediction: str, gold: str) -> float:
    """Lenient metric: the gold answer appears inside the prediction."""
    return float(normalize_answer(gold) in normalize_answer(prediction))


def evidence_in_text(evidence: str, text: str) -> bool:
    """A retrieved chunk is relevant if it contains the gold evidence phrase."""
    return " ".join(evidence.lower().split()) in " ".join(text.lower().split())


def reciprocal_rank(rank: Optional[int]) -> float:
    return 1.0 / rank if rank else 0.0
