import json, re
import os, math, time, json, random
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

BATCH_SIZE = 32
LLM_MAX_TOKENS = 16      # ตอบเป็น JSON array สั้น ๆ
LLM_TEMPERATURE = 0.0

LABEL_SET = ["negative", "neutral", "positive"]
USE_OLLAMA = True 
# OLLAMA_MODEL = "llama3.1" 
OLLAMA_MODEL = "llama3.1:latest"   # เดิม: "llama3.1"

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
LABEL_SET = ["negative", "neutral", "positive"]

RETRY = 2


def normalize_label(s):
    s = (s or "").strip().lower()
    # keep only one of LABEL_SET (robust to punctuation)
    for cand in LABEL_SET:
        if cand in s:
            return cand
    # fallback: unknown -> neutral
    return "neutral"

def llm_chat(messages, temperature=0.0, max_tokens=8):
    if USE_OLLAMA:
        return chat_completion_ollama(messages, temperature, max_tokens)
    else:
        # return chat_completion_openai(messages, temperature, max_tokens)
        return None



def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def build_batch_prompt(texts, shots=None):
    """
    บังคับให้โมเดลส่งคืน ONLY JSON array ของ labels
    ความยาวต้องเท่าจำนวน texts ที่ส่งเข้าไป
    """
    sys = (
        "You are a precise sentiment classifier.\n"
        "Return ONLY a JSON array of strings, one per input, no extra text.\n"
        "Each string must be exactly one of: \"negative\", \"neutral\", \"positive\"."
    )
    messages = [{"role": "system", "content": sys}]
    if shots:
        for ex_text, ex_label in shots:
            messages.append({"role": "user", "content": f"Text: {ex_text}\nLabel:"})
            messages.append({"role": "assistant", "content": ex_label})

    payload_lines = [f"{i+1}. {t}" for i, t in enumerate(texts)]
    user = (
        "Classify each line and return a JSON array of labels.\n"
        "INPUTS:\n" + "\n".join(payload_lines) +
        "\n\nReturn JSON array only (e.g., [\"neutral\",\"positive\",...])."
    )
    messages.append({"role": "user", "content": user})
    return messages

def parse_labels_from_json(s, n_expected):
    s = (s or "").strip()
    # กันกรณีโมเดลเผลอพิมพ์อย่างอื่นเพิ่ม: ดึงเฉพาะ JSON array
    m = re.search(r'\[.*\]', s, flags=re.S)
    if m:
        s = m.group(0)
    try:
        arr = json.loads(s)
        if isinstance(arr, list) and len(arr) == n_expected:
            return [normalize_label(x) for x in arr]
    except Exception:
        pass
    # ถ้า parse ไม่ได้ ให้คืน neutral ทั้งชุด (กันตก)
    return ["neutral"] * n_expected

def evaluate_split_batched(ds_split, shots=None, max_items=None, name=""):
    # เตรียมข้อมูล
    data = ds_split if max_items is None else ds_split.select(range(max_items))
    texts = [r["text"] for r in data]
    golds = [int(r["label"]) for r in data]

    y_pred = []
    total = len(texts)
    for bi, start in enumerate(range(0, total, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, total)
        batch_texts = texts[start:end]

        if bi % 5 == 0:
            print(f"[{name}] batch {bi} ({start}:{end}/{total})", flush=True)

        msgs = build_batch_prompt(batch_texts, shots=shots)
        out = llm_chat(msgs, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
        labels = parse_labels_from_json(out, n_expected=len(batch_texts))
        y_pred.extend([LABEL2ID.get(lab, 1) for lab in labels])

    acc = accuracy_score(golds, y_pred)
    f1m = f1_score(golds, y_pred, average="macro")
    return {"accuracy": acc, "f1_macro": f1m}

def chat_completion_ollama(messages, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS):
    import requests, time
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": OLLAMA_MODEL,             # เช่น "llama3.1:latest" หรือ "gemma3:latest"
        "messages": messages,
        "format": "json",   
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            # "num_ctx": 2048,             # เพิ่ม context ถ้า batch ยาวมาก
        },
        "stream": False,
    }
    for i in range(RETRY):
        try:
            r = requests.post(
                url, json=payload, timeout=30,
                proxies={"http": None, "https": None}
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            if i == RETRY - 1:
                raise
            time.sleep(1.0 * (i + 1))
            
def make_shots(dataset, n=1, seed=42):
    """
    เลือกตัวอย่างจาก train มาเป็น in-context examples
    พยายามให้กระจายครบ 3 คลาส
    """
    random.seed(seed)
    by_label = {0: [], 1: [], 2: []}
    for r in dataset:
        lab = int(r["label"])
        if len(by_label[lab]) < math.ceil(n/3):
            by_label[lab].append((r["text"], ID2LABEL[lab]))
        if sum(len(v) for v in by_label.values()) >= n:
            break
    # ถ้ายังไม่ครบ n เติมแบบสุ่ม
    all_rows = list(dataset)
    while sum(len(v) for v in by_label.values()) < n:
        rr = random.choice(all_rows)
        by_label[int(rr["label"])].append((rr["text"], ID2LABEL[int(rr["label"])]))
    shots = []
    for lab in [0,1,2]:
        shots.extend(by_label[lab][: math.ceil(n/3)])
    return shots[:n]

# print("\n== ZERO-SHOT (BATCH x32) ==")
# zs_val = evaluate_split_batched(val, shots=None, max_items=None, name="ZS/val-batch")
# print({"split": "validation", **zs_val})

if __name__ == "__main__":
    
    # print("Loading dataset…")
    # ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    # val = ds["validation"]
    # test = ds["test"]
    
    # print("\n== ZERO-SHOT (BATCH x32) ==")
    # zs_val = evaluate_split_batched(val, shots=None, max_items=None, name="ZS/val-batch")
    # print({"split": "validation", **zs_val})
    
    # train = ds["train"]

    # # === ONE-SHOT ===
    # print("\n== ONE-SHOT (BATCH x32) ==")
    # one_shot = make_shots(train, n=1)
    # os_val = evaluate_split_batched(val, shots=one_shot, max_items=None, name="ONE/val-batch")
    # print({"split": "validation", **os_val})

    # # === FEW-SHOT (เช่น 6 ตัวอย่าง) ===
    # print("\n== FEW-SHOT (n=6, BATCH x32) ==")
    # few_shots = make_shots(train, n=6)
    # fs_val = evaluate_split_batched(val, shots=few_shots, max_items=None, name="FEW/val-batch")
    # print({"split": "validation", **fs_val})
    ma = "I love it"
    print(chat_completion_ollama(ma))

