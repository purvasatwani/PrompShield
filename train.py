"""Train + evaluate. Model selection uses 5-fold CV on TRAIN only; the TEST set is touched once for the final report.
Run: python train.py          (add --no-embed to skip the embedding model, e.g. low RAM)"""
import json
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline, make_union
from common import EMBED_MODEL, REPORTS, STORE, embed, load_data


def tfidf_model():
    return make_pipeline(
        make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        LogisticRegression(max_iter=2000, C=5, class_weight="balanced"))


def embed_model():
    return LogisticRegression(max_iter=2000, C=2, class_weight="balanced")


def pick_threshold(y, p, min_recall=0.95):
    """Highest threshold that still catches >= 95% of attacks (in a firewall, missed attacks cost more)."""
    _, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= min_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"threshold": thr, "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p)}


def main():
    tr, te = load_data()
    Xtr_t, Xte_t = tr.text.astype(str).tolist(), te.text.astype(str).tolist()
    ytr, yte = tr.label.values, te.label.values
    setups = {"tfidf_lr": (Xtr_t, Xte_t, tfidf_model)}
    if "--no-embed" not in sys.argv:
        print("Embedding texts (first run downloads the model)...")
        setups["embed_lr"] = (embed(Xtr_t), embed(Xte_t), embed_model)

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    results, errors, fitted = {}, [], {}
    fig, axs = plt.subplots(1, len(setups), figsize=(4.5 * len(setups), 4), squeeze=False)
    for ax, (name, (Xa, Xb, factory)) in zip(axs[0], setups.items()):
        cv_f1 = cross_val_score(factory(), Xa, ytr, cv=skf, scoring="f1")
        oof = cross_val_predict(factory(), Xa, ytr, cv=skf, method="predict_proba")[:, 1]
        thr = pick_threshold(ytr, oof)
        model = factory().fit(Xa, ytr)
        p = model.predict_proba(Xb)[:, 1]
        results[name] = {"cv_f1_mean": cv_f1.mean(), "cv_f1_std": cv_f1.std(),
                         "test_at_0.5": metrics(yte, p, 0.5), "test_at_tuned_threshold": metrics(yte, p, thr)}
        fitted[name] = (model, thr)
        pred = (p >= thr).astype(int)
        ConfusionMatrixDisplay.from_predictions(yte, pred, ax=ax, colorbar=False, display_labels=["legit", "attack"])
        ax.set_title(f"{name} (thr={thr:.2f})")
        for i in np.where(pred != yte)[0]:
            errors.append({"model": name, "true": int(yte[i]), "pred": int(pred[i]), "score": round(float(p[i]), 3), "text": Xte_t[i]})
        r = results[name]
        print(f"\n{name}: CV F1 {r['cv_f1_mean']:.3f}±{r['cv_f1_std']:.3f}")
        for k in ("test_at_0.5", "test_at_tuned_threshold"):
            m = r[k]
            print(f"  {k}: acc {m['accuracy']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} AUC {m['roc_auc']:.3f}")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrices.png", dpi=120)

    best = max(results, key=lambda n: results[n]["cv_f1_mean"])  # chosen by CV, not by test
    model, thr = fitted[best]
    joblib.dump({"kind": best, "model": model, "threshold": thr, "embed_model": EMBED_MODEL}, STORE / "guard.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps({"best_model": best, "results": results}, indent=2), encoding="utf-8")
    pd.DataFrame(errors).to_csv(REPORTS / "errors.csv", index=False)
    print(f"\nBest by CV: {best}. Saved storage/guard.joblib, reports/metrics.json, reports/errors.csv")


if __name__ == "__main__":
    main()
