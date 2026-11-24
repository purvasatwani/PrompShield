# PromptShield - prompt injection firewall

## 1. Install (Windows, Python 3.10+)
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

First run downloads: dataset `deepset/prompt-injections` (~small) and the embedding model (~470 MB). Needs internet once.
If Hugging Face is blocked: open https://huggingface.co/datasets/deepset/prompt-injections , download the train/test
files, save as data/train.csv and data/test.csv with columns `text,label`.

## 2. Scan the dataset
    python scan_dataset.py        -> reports/dataset_scan.md, reports/dataset_charts.png

## 3. Train + test
    python train.py               (python train.py --no-embed if low on RAM)
    -> reports/metrics.json, confusion_matrices.png, errors.csv, storage/guard.joblib

## 4. Red-team test
    python attack_test.py         -> reports/attack_test.md

## 5. Demo
    streamlit run app.py
