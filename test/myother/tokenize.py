from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_and_print(ex):
    print("====== NEW BATCH ======")
    print(ex)   # ดูว่า ex เป็นยังไง
    return tok(ex["text"], truncation=True)

# ลองแค่ validation split เพื่อไม่ให้ print เยอะเกิน
ds_enc = ds["validation"].map(tokenize_and_print, batched=True, batch_size=3)
