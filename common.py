import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA, REPORTS, STORE = ROOT / "data", ROOT / "reports", ROOT / "storage"
for d in (DATA, REPORTS, STORE):
    d.mkdir(exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_embedder = None


def load_data():
    """Returns (train_df, test_df) with columns text,label. label 1 = injection, 0 = legitimate.
    First run downloads deepset/prompt-injections and caches it as CSV in data/."""
    tr, te = DATA / "train.csv", DATA / "test.csv"
    if not tr.exists() or not te.exists():
        from datasets import load_dataset
        ds = load_dataset("deepset/prompt-injections")
        ds["train"].to_pandas().to_csv(tr, index=False)
        ds["test"].to_pandas().to_csv(te, index=False)
    return pd.read_csv(tr), pd.read_csv(te)


def embed(texts):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
