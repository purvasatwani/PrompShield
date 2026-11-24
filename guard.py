"""The firewall. Hybrid: ML score + regex rules."""
import re
import joblib
from common import STORE, embed

RULES = {
    "ignore-previous": r"\b(ignore|disregard|forget)\b.{0,25}\b(previous|prior|above|earlier|all)\b.{0,25}\b(instructions?|prompts?|rules?|everything)\b",
    "reveal-prompt": r"\b(reveal|print|show|output|repeat)\b.{0,30}\b(system prompt|hidden instructions?|initial instructions?|your instructions)\b",
    "persona-jailbreak": r"\b(you are now|pretend (that )?you|act as)\b.{0,40}\b(dan|no (safety|restrictions|rules)|unrestricted|developer mode)\b",
    "override": r"\b(system override|developer mode|bypass (all )?(content )?(policies|filters|safety))\b",
    "german": r"\b(ignoriere|vergiss)\b.{0,30}\b(alle|anweisungen|vorherigen)\b",
}


class Guard:
    def __init__(self):
        b = joblib.load(STORE / "guard.joblib")
        self.kind, self.model, self.threshold = b["kind"], b["model"], b["threshold"]

    def score(self, text):
        x = embed([text]) if self.kind == "embed_lr" else [text]
        return float(self.model.predict_proba(x)[0, 1])

    @staticmethod
    def rules(text):
        return [n for n, p in RULES.items() if re.search(p, text, re.I | re.S)]

    def check(self, text, use_rules=True, threshold=None):
        s, hit = self.score(text), self.rules(text)
        thr = self.threshold if threshold is None else threshold
        return {"score": s, "threshold": thr, "rules": hit,
                "blocked": s >= thr or (use_rules and bool(hit))}
