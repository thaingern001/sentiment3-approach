print("Loading dataset…")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    val = ds["validation"]
    test = ds["test"]
    
    print("\n== ZERO-SHOT (BATCH x32) ==")
    zs_val = evaluate_split_batched(val, shots=None, max_items=None, name="ZS/val-batch")
    print({"split": "validation", **zs_val})