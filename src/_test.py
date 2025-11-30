from transformers import pipeline
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", use_safetensors=True)
clf = pipeline("sentiment-analysis", model=model)
clf("I love this!")  # ได้ positive/neutral/negative แม่นกว่า