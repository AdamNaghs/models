from datasets import load_dataset

USER_TAG = "<|user|>"
ASSIST_TAG = "<|assistant|>"
END_TAG = "<|end|>"

# Start smaller for quick iteration; set None to use full split.
MAX_EXAMPLES = 20000
OUTPUT_PATH = "data.txt"


def clean(text: str) -> str:
    return " ".join((text or "").split()).strip()


def to_stinker(messages):
    out = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = clean(m.get("content") or "")
        if not content:
            continue
        if role == "user":
            out.append(f"{USER_TAG} {content}\n")
        elif role == "assistant":
            out.append(f"{ASSIST_TAG} {content} {END_TAG}\n")
    s = "".join(out)
    return s if (USER_TAG in s and ASSIST_TAG in s) else ""


def main():
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    if MAX_EXAMPLES is not None:
        ds = ds.select(range(min(MAX_EXAMPLES, len(ds))))

    kept = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in ds:
            s = to_stinker(ex.get("messages", []))
            if s:
                f.write(s + "\n")
                kept += 1

    print(f"Wrote {OUTPUT_PATH} with {kept} conversations")


if __name__ == "__main__":
    main()
