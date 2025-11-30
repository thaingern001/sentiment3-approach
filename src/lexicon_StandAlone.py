# pip install datasets nltk scikit-learn
import nltk
nltk.download("opinion_lexicon")
from nltk.corpus import opinion_lexicon

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

pos_words = set(opinion_lexicon.positive())
neg_words = set(opinion_lexicon.negative())
negations = {"not","no","never","n't"}

def lexicon_score(text):
    toks = text.lower().split()
    s, i = 0, 0
    while i < len(toks):
        w = toks[i]
        if w in negations and i + 1 < len(toks):
            nxt = toks[i+1]
            if nxt in pos_words: s -= 1
            elif nxt in neg_words: s += 1
            i += 2
            continue
        if w in pos_words: s += 1
        if w in neg_words: s -= 1
        i += 1
    return s

def score_to_label(s, pos_th=1, neg_th=-1):
    if s >= pos_th: return 2
    if s <= neg_th: return 0
    return 1

def lexicon_model(msg):
    return score_to_label(lexicon_score(msg))

ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

def eval_split(split_name="validation"):
    y_true, y_pred = [], []
    for r in ds[split_name]:
        y_true.append(int(r["label"]))
        sc = lexicon_score(r["text"])
        y_pred.append(score_to_label(sc))
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    return {"split": split_name, "accuracy": acc, "f1_macro": f1}

# print(eval_split("validation"))
# print(eval_split("test"))

