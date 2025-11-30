from pathlib import Path
import pandas as pd

class LexiconModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.text_col = cfg["dataset"]["text_col"]
        self.label_col = cfg["dataset"].get("label_col")
        # โหลด opinion lexicon (ใส่ไฟล์ไว้ใน data/external/)
        self.pos = self._load_set("data/external/positive-words.txt")
        self.neg = self._load_set("data/external/negative-words.txt")
        self.negations = {"not","no","never","n't","ไม่","ไม่ได้","ไม่มี"}

    def _load_set(self, path):
        p = Path(path)
        if not p.exists(): return set()
        return {w.strip() for w in p.read_text(encoding="utf-8", errors="ignore").splitlines()
                if w and not w.startswith(";")}

    def _score_tokens(self, toks):
        s = 0
        i = 0
        while i < len(toks):
            w = toks[i].lower()
            if w in self.negations and i+1 < len(toks):
                nxt = toks[i+1].lower()
                if nxt in self.pos: s -= 1
                elif nxt in self.neg: s += 1
                i += 2
                continue
            if w in self.pos: s += 1
            if w in self.neg: s -= 1
            i += 1
        return s

    def predict_text(self, text: str):
        toks = str(text).split()
        sc = self._score_tokens(toks)
        if sc >= self.cfg["lexicon"]["threshold_pos"]: return {"label":"positive","score":sc}
        if sc <= self.cfg["lexicon"]["threshold_neg"]: return {"label":"negative","score":sc}
        return {"label":"neutral","score":sc}

    # สำหรับงานที่มี label
    def _eval_on(self, df):
        from sklearn.metrics import accuracy_score, f1_score
        preds = df[self.text_col].map(lambda t: self.predict_text(t)["label"])
        acc = accuracy_score(df[self.label_col], preds)
        f1  = f1_score(df[self.label_col], preds, average="macro")
        print({"accuracy":acc, "f1_macro":f1})

    def train(self):
        # lexicon ไม่ต้องเทรน แค่ยืนยันไฟล์พร้อมใช้งาน
        print("Lexicon ready. (no training)")

    def evaluate(self):
        if not self.label_col:
            print("No label column configured.")
            return
        df = pd.read_csv("data/processed/test.csv")
        self._eval_on(df)
