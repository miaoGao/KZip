"""Exact-Match and token-level F1, following the MRQA / HotpotQA convention."""

import re
import string
from collections import Counter


def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(results):
    """`results`: list of {"pred_ans", "answer"}. Returns averaged EM and F1."""
    if not results:
        return {"EM": 0.0, "F1": 0.0, "n": 0}
    em = sum(exact_match(r["pred_ans"], r["answer"]) for r in results) / len(results)
    f1_score = sum(f1(r["pred_ans"], r["answer"]) for r in results) / len(results)
    return {"EM": round(em, 4), "F1": round(f1_score, 4), "n": len(results)}
