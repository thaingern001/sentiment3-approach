import argparse, yaml, random, numpy as np
from src.dataio import prepare_data
from src.lexicon import LexiconModel
from src.classical import ClassicalModel

def set_seed(s):
    random.seed(s); np.random.seed(s)

def main():
    ap = argparse.ArgumentParser("textsentiment-lite")
    ap.add_argument("cmd", choices=["prepare", "train", "eval", "predict"])
    ap.add_argument("--model", default="lexicon", choices=["lexicon","classical"])
    ap.add_argument("--text", default=None)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 42))

    if args.cmd == "prepare":
        prepare_data(cfg)
        print("✅ prepared -> data/processed/{train,valid,test}.csv")
        return

    if args.model == "lexicon":
        m = LexiconModel(cfg)
    else:
        m = ClassicalModel(cfg)

    if args.cmd == "train":
        m.train()
    elif args.cmd == "eval":
        m.evaluate()
    elif args.cmd == "predict":
        assert args.text, "--text required"
        print(m.predict_text(args.text))

if __name__ == "__main__":
    main()
