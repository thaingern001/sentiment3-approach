from datasets import load_dataset
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
from transformers import (
    AutoTokenizer,
    set_seed,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    )

# def eval_split(split_name="validation"):
#     y_true, y_pred = [], []
#     for r in ds[split_name]:
#         y_true.append(int(r["label"]))
#         sc = lexicon_score(r["text"])
#         y_pred.append(score_to_label(sc))
#     acc = accuracy_score(y_true, y_pred)
#     f1  = f1_score(y_true, y_pred, average="macro")
#     return {"split": split_name, "accuracy": acc, "f1_macro": f1}


ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

seed = 42 #can change
set_seed(seed) # just for controll final score to stable
# model_name = "distilbert-base-uncased" #why use this model? ==> DistilBERT for Normal use
model_name = "cardiffnlp/twitter-roberta-base-sentiment" #more perfomance for this
tok = AutoTokenizer.from_pretrained(model_name)

'''
ex = {
  "text": ["I love this!", "I hate that!"],
  "label": [2, 0]
}
'''
def tokenize(ex): #batch --encode--> encoded batch
    return tok(ex["text"], truncation=True)
'''
{ return
  'input_ids': [
      [101, 1045, 2293, 2023, 999, 102],
      [101, 1045, 5223, 2008, 999, 102]
  ],
  'attention_mask': [
      [1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1]
  ]
}'''

ds_enc = ds.map(tokenize, batched=True)

    
# ex = {
#   "text": ["I love this!", "I hate that!"],
#   "label": [2, 0]
# }

# print(tokenize(ex))

collator = DataCollatorWithPadding(tokenizer=tok) #ปรับผลจากการTokenizeให้ยาวเท่า by padding --> ขยายออก

id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=model_name,
    num_labels = 3,
    label2id=label2id,
    id2label=id2label
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred # logits = softmax([-1.2, 0.5, 1.8]) 
    preds = np.argmax(logits, axis=-1) # softmax([-1.2, 0.5, 1.8]) ==> index2 is max ==> preds=2
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

args = TrainingArguments( #===
    output_dir="./out",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    num_train_epochs=2,
    # evaluation_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    report_to="none",
    seed=42,
)

#จุดรวม
trainer = Trainer( #===
    model=model,
    args=args,
    train_dataset=ds_enc["train"],
    eval_dataset=ds_enc["validation"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()

def eval_split(split_name="validation"):
    pred = trainer.predict(ds_enc[split_name])
    metrics = compute_metrics((pred.predictions, pred.label_ids))
    return {"split": split_name, "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"]}

print(eval_split("validation"))
print(eval_split("test"))