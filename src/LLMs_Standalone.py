import os, math, time, json, random
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# ====== SETTINGS 
SEED = 42
random.seed(SEED)

# Choose backend
USE_OLLAMA = True  # True = use Ollama local server, False = use OpenAI
# OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-4.1-mini"
OLLAMA_MODEL = "llama3.1"     # a common local instruct model

TPS_LIMIT = 0    # target ~1.5 prompts/sec
RETRY = 2

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
LABEL_SET = ["negative", "neutral", "positive"]


# ===== LLM Clients =====  #
'''
def chat_completion_openai(messages, temperature=0.0, max_tokens=8):
    """
    Minimal OpenAI Chat API call. Requires OPENAI_API_KEY in env.
    """
    from openai import OpenAI
    client = OpenAI()
    for i in range(RETRY):
        try:
            rsp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            if i == RETRY - 1:
                raise
            time.sleep(1.5 * (i + 1))
'''

def chat_completion_ollama(messages, temperature=0.0, max_tokens=8):
    """
    Simple Ollama chat call via HTTP to localhost.
    Make sure: `ollama serve` and model pulled: `ollama pull llama3.1`
    """
    import requests
    # url = "http://localhost:11434/api/chat"
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    for i in range(RETRY):
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"].strip()
        except Exception as e:
            if i == RETRY - 1:
                raise
            time.sleep(1.5 * (i + 1))
def llm_chat(messages, temperature=0.0, max_tokens=8):
    if USE_OLLAMA:
        return chat_completion_ollama(messages, temperature, max_tokens)
    else:
        # return chat_completion_openai(messages, temperature, max_tokens)
        return None
    
    
    
# ===== Prompt Utils ===== #
SYSTEM_PROMPT = (
    "You are a precise sentiment classifier. "
    "Given a single English tweet text, respond with exactly one of: "
    "'negative', 'neutral', or 'positive'. Do not add explanations."
)


'''
shots = [
    ("I hate that!", "negative"),
    ("Meh, it’s okay.", "neutral"),
    ("I love this!", "positive")
]
'''
def build_user_prompt(text, shots=None):
    """
    shots: list of (example_text, label_string) or None
    Returns a chat-format messages list with optional in-context examples.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # add few-shot examples (as user/assistant turns)
    if shots:
        for ex_text, ex_label in shots:
            messages.append({"role": "user", "content": f"Text: {ex_text}\nLabel:"})
            messages.append({"role": "assistant", "content": ex_label})
    # the actual instance
    messages.append({"role": "user", "content": f"Text: {text}\nLabel:"})
    return messages


"""
normalize_label("Positive")          # "positive"
normalize_label("NEGATIVE")          # "negative"
normalize_label("This is Neutral.")  # "neutral"
normalize_label("maybe pos")         # "positive"  (เพราะ "pos" มี "positive"? -> ถ้าอยาก strict ต้องใช้ regex)
normalize_label(None)                # "neutral"
normalize_label("angry")             # "neutral" (fallback)
"""
def normalize_label(s):
    s = (s or "").strip().lower()
    # keep only one of LABEL_SET (robust to punctuation)
    for cand in LABEL_SET:
        if cand in s:
            return cand
    # fallback: unknown -> neutral
    return "neutral"


# ====== Evaluator ======  #
'''
temperature
0.0 → deterministic (ตอบเหมือนเดิมทุกครั้ง, เน้นแม่นยำ ไม่มั่ว)
~0.7 → ปกติ (สมดุลระหว่างความหลากหลายกับความแม่นยำ)
>1.0 → สุ่มเยอะขึ้น อาจสร้างคำใหม่ ๆ ที่คาดไม่ถึง แต่ก็เสี่ยงมั่ว

tokens
unit of tokens
'''

def evaluate_split(ds_split, shots=None, max_items=None, tps=TPS_LIMIT):
    y_true, y_pred = [], []
    delay = 1.0 / tps if tps > 0 else 0.0
    
    # print("#debug 1")

    it = ds_split if max_items is None else ds_split.select(range(max_items))
    for r in it:
        # print("HAHAHAHAH")
        text = r["text"]
        gold = int(r["label"])
        msgs = build_user_prompt(text, shots=shots)
        out = llm_chat(msgs, temperature=0.0, max_tokens=4)
        pred_label = normalize_label(out)
        y_true.append(gold)
        y_pred.append(LABEL2ID.get(pred_label, 1))  # default neutral
        print(pred_label,gold,text)
        # time.sleep(delay)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1_macro": f1m}

# ====== Few-shot set ==== #
def make_shots(dataset, n=4):
    """
    Build few-shot examples balanced across classes if possible.
    """
    by_label = {0: [], 1: [], 2: []}
    for r in dataset:
        lab = int(r["label"])
        if len(by_label[lab]) < math.ceil(n / 3):
            by_label[lab].append((r["text"], ID2LABEL[lab]))
        if sum(len(v) for v in by_label.values()) >= n:
            break
    # if not balanced, just top-up randomly
    all_rows = list(dataset)
    while sum(len(v) for v in by_label.values()) < n:
        rr = random.choice(all_rows)
        by_label[int(rr["label"])].append((rr["text"], ID2LABEL[int(rr["label"])]))
    shots = []
    for lab in [0, 1, 2]:
        shots.extend(by_label[lab][: math.ceil(n / 3)])
    return shots[:n]


# RUN

if __name__ == "__main__":
    print("Loading dataset…")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    val = ds["validation"]
    test = ds["test"]

    # ---- ZERO-SHOT ----
    # print("\n== ZERO-SHOT ==")
    # zs_val = evaluate_split(val, shots=None, max_items=None)
    # zs_test = evaluate_split(test, shots=None, max_items=None)
    # print({"split": "validation", **zs_val})
    # print({"split": "test", **zs_test})

    # # ---- ONE-SHOT ----
    # print("\n== ONE-SHOT ==")
    # one_shot = make_shots(ds["train"], n=1)
    # os_val = evaluate_split(val, shots=one_shot, max_items=None)
    # os_test = evaluate_split(test, shots=one_shot, max_items=None)
    # print({"split": "validation", **os_val})
    # print({"split": "test", **os_test})

    # ---- FEW-SHOT (e.g., 6) ----
    print("\n== FEW-SHOT (6) ==")
    few_shots = make_shots(ds["train"], n=6)
    fs_val = evaluate_split(val, shots=few_shots, max_items=None)
    fs_test = evaluate_split(test, shots=few_shots, max_items=None)
    print({"split": "validation", **fs_val})
    print({"split": "test", **fs_test})
    
    print("End ")