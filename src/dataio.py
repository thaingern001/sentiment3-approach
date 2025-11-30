import re, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def _clean(s: str) -> str:
    s = re.sub(r'https?://\S+|@\w+|#\w+', ' ', str(s))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def prepare_data(cfg):
    p = cfg["dataset"]["path"]
    text_col = cfg["dataset"]["text_col"]
    label_col = cfg["dataset"].get("label_col")

    df = pd.read_csv(p)
    df[text_col] = df[text_col].map(_clean)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    tr_ratio = cfg["dataset"]["train_ratio"]
    va_ratio = cfg["dataset"]["valid_ratio"]

    if label_col and label_col in df.columns:
        tr, rest = train_test_split(df, train_size=tr_ratio, random_state=42, stratify=df[label_col])
        va_size = va_ratio/(1-tr_ratio)
        va, te = train_test_split(rest, test_size=1-va_size, random_state=42, stratify=rest[label_col])
    else:
        tr, rest = train_test_split(df, train_size=tr_ratio, random_state=42)
        va_size = va_ratio/(1-tr_ratio)
        va, te = train_test_split(rest, test_size=1-va_size, random_state=42)

    tr.to_csv("data/processed/train.csv", index=False)
    va.to_csv("data/processed/valid.csv", index=False)
    te.to_csv("data/processed/test.csv",  index=False)
