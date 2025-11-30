import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import joblib

class ClassicalModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.text_col = cfg["dataset"]["text_col"]
        self.label_col = cfg["dataset"]["label_col"]
        model_name = cfg["classical"]["model"]
        clf = LogisticRegression(max_iter=200) if model_name=="logreg" else LinearSVC()
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=cfg["classical"]["max_features"])),
            ("clf", clf),
        ])

    def train(self):
        tr = pd.read_csv("data/processed/train.csv")
        va = pd.read_csv("data/processed/valid.csv")
        self.pipe.fit(tr[self.text_col], tr[self.label_col])
        # quick val
        preds = self.pipe.predict(va[self.text_col])
        acc = accuracy_score(va[self.label_col], preds)
        f1  = f1_score(va[self.label_col], preds, average="macro")
        print({"val_accuracy":acc, "val_f1_macro":f1})
        joblib.dump(self.pipe, "model_classical.joblib")

    def evaluate(self):
        te = pd.read_csv("data/processed/test.csv")
        self.pipe = joblib.load("model_classical.joblib")
        preds = self.pipe.predict(te[self.text_col])
        acc = accuracy_score(te[self.label_col], preds)
        f1  = f1_score(te[self.label_col], preds, average="macro")
        print({"test_accuracy":acc, "test_f1_macro":f1})

    def predict_text(self, text:str):
        if hasattr(self.pipe, "predict"):
            return {"label": self.pipe.predict([text])[0]}
        else:
            from joblib import load
            self.pipe = load("model_classical.joblib")
            return {"label": self.pipe.predict([text])[0]}
