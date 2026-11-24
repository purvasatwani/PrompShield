"""Scan the dataset for problems BEFORE training. Output: reports/dataset_scan.md + 2 charts.
Run: python scan_dataset.py"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from common import REPORTS, load_data

GERMAN = {"der", "das", "und", "ist", "nicht", "ich", "ein", "eine", "den", "mit", "von", "sie",
          "wie", "auf", "für", "wer", "zu", "dem", "des", "auch", "bitte", "wir", "es"}
out = []


def sec(title, body):
    out.append(f"## {title}\n\n{body}\n")
    print(f"\n=== {title} ===\n{body}")


def is_german(t):  # rough heuristic only
    return sum(w in GERMAN for w in re.findall(r"\w+", t.lower())) >= 2


tr, te = load_data()
tr["split"], te["split"] = "train", "test"
both = pd.concat([tr, te], ignore_index=True)
both["text"] = both["text"].fillna("").astype(str)
both["chars"] = both.text.str.len()
both["words"] = both.text.str.split().str.len()
both["german"] = both.text.map(is_german)
both["nonascii"] = both.text.map(lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1))

# 1. basics
sec("1. Size, columns, missing values",
    f"- train rows: {len(tr)}, test rows: {len(te)}\n- columns: {list(tr.columns)}\n"
    f"- empty/missing texts: {int((both.text.str.strip() == '').sum())}\n"
    f"- labels present: {sorted(both.label.unique().tolist())} (1 = injection)")

# 2. class balance
bal = both.groupby(["split", "label"]).size().unstack(fill_value=0)
bal["injection_%"] = (bal[1] / bal.sum(axis=1) * 100).round(1)
sec("2. Class balance", "```\n" + bal.to_string() + "\n```\n"
    "If one class is far bigger, accuracy is misleading: use precision/recall/F1 instead.")

# 3. length
ln = both.groupby("label")[["chars", "words"]].describe().round(1)
sec("3. Text length by label", "```\n" + both.groupby("label")[["chars", "words"]].agg(["mean", "median", "max"]).round(1).to_string() + "\n```")

# 4. duplicates + leakage
dup_tr = int(tr.text.duplicated().sum())
overlap = set(tr.text) & set(te.text)
tfv = TfidfVectorizer(ngram_range=(1, 2)).fit(both.text)
sim = cosine_similarity(tfv.transform(te.text.astype(str)), tfv.transform(tr.text.astype(str))).max(axis=1)
near = np.where(sim > 0.9)[0]
ex = "\n".join(f"  - sim={sim[i]:.2f}: {te.text.iloc[i][:90]!r}" for i in near[:5])
sec("4. Duplicates and train/test leakage",
    f"- exact duplicates inside train: {dup_tr}\n- exact text shared by train and test: {len(overlap)}\n"
    f"- test rows with a near-duplicate (cosine > 0.9) in train: {len(near)} of {len(te)}\n{ex}\n\n"
    "Leakage inflates your test score. If many rows leak, remove them from test before reporting results.")

# 5. language shortcut
lang = both.groupby("label")["german"].mean().mul(100).round(1)
sec("5. Language mix (heuristic)", f"% of texts that look German, by label:\n```\n{lang.to_string()}\n```\n"
    "If one class is much more German than the other, a model can cheat by detecting language, not attacks.")

# 6. discriminative words
cv = CountVectorizer(min_df=3, binary=True, lowercase=True)
Xb = cv.fit_transform(tr.text.astype(str))
y = tr.label.values
c1, c0 = np.asarray(Xb[y == 1].sum(0)).ravel(), np.asarray(Xb[y == 0].sum(0)).ravel()
lo = np.log((c1 + 1) / ((y == 1).sum() + 2)) - np.log((c0 + 1) / ((y == 0).sum() + 2))
vocab = np.array(cv.get_feature_names_out())
top1, top0 = vocab[np.argsort(-lo)[:15]], vocab[np.argsort(lo)[:15]]
sec("6. Words most tied to each class (log-odds)",
    f"- injection: {', '.join(top1)}\n- legitimate: {', '.join(top0)}\n\n"
    "Words like 'ignore' or 'instructions' are expected. Odd words (names, topics) mean the dataset has shortcuts.")

# 7. shortcut baseline
feats = both[both.split == "train"][["chars", "words", "german", "nonascii"]].astype(float).values
sc = cross_val_score(LogisticRegression(max_iter=1000), feats, y, cv=5, scoring="accuracy").mean()
major = max(y.mean(), 1 - y.mean())
sec("7. Shortcut check (no real language understanding)",
    f"- majority-class accuracy: {major:.3f}\n- accuracy using ONLY length/language features: {sc:.3f}\n\n"
    "If the second number is well above the first, those superficial features leak the label. "
    "Your real model must beat this to prove it learned something.")

# charts
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
bal[[0, 1]].plot.bar(ax=ax[0], title="Class balance")
for lab, g in both.groupby("label"):
    ax[1].hist(g.chars.clip(upper=600), bins=30, alpha=0.6, label=f"label {lab}")
ax[1].set_title("Text length (chars, clipped at 600)")
ax[1].legend()
plt.tight_layout()
plt.savefig(REPORTS / "dataset_charts.png", dpi=120)

(REPORTS / "dataset_scan.md").write_text("# Dataset scan\n\n" + "\n".join(out), encoding="utf-8")
print(f"\nSaved reports/dataset_scan.md and reports/dataset_charts.png")
