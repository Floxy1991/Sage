"""Build ESConv train/valid/test examples for the SAGE reproduction.

Each example is one supporter turn: the dialogue context before it, the gold
response and its strategy. Two splits are produced:

  paper  - 910/130/260 dialogues (7:1:2), as stated in SAGE Sec. 4.1. The paper
           gives no seed, so we shuffle with seed 13 (the seed of the common split).
  zcj    - 70/15/15, the split from thu-coai codes_zcj/_reformat/process.py
           (random.seed(13)), used by most prior ESC baselines.

Usage: python prepare_data.py --esconv data/ESConv.json --out data/
"""
import argparse
import json
import os
import random

STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]


def norm(text):
    return " ".join(text.strip().split())


def reformat(d):
    # Mirrors codes_zcj process_data, so the zcj split reproduces exactly.
    dial = []
    for u in d["dialog"]:
        turn = {"text": norm(u["content"]), "speaker": "usr" if u["speaker"] == "seeker" else "sys"}
        if turn["speaker"] == "sys":
            turn["strategy"] = u["annotation"]["strategy"]
        dial.append(turn)
    return {
        "emotion_type": d["emotion_type"],
        "problem_type": d["problem_type"],
        "situation": d["situation"],
        "dialog": dial,
    }


def split_dialogs(data, mode):
    data = list(data)
    random.seed(13)
    random.shuffle(data)
    n = len(data)
    if mode == "zcj":
        dev, test = int(0.15 * n), int(0.15 * n)
        return {"valid": data[:dev], "test": data[dev:dev + test], "train": data[dev + test:]}
    train, dev = round(0.7 * n), round(0.1 * n)
    return {"train": data[:train], "valid": data[train:train + dev], "test": data[train + dev:]}


def to_examples(dialogs, split):
    out = []
    for di, d in enumerate(dialogs):
        for ti, turn in enumerate(d["dialog"]):
            # Skip opening supporter turns with no seeker utterance before them.
            if turn["speaker"] != "sys" or not any(t["speaker"] == "usr" for t in d["dialog"][:ti]):
                continue
            ctx = d["dialog"][:ti]
            last_usr = next(t["text"] for t in reversed(ctx) if t["speaker"] == "usr")
            out.append({
                "id": f"{split}-{di}-{ti}",
                "dialog_idx": di,
                "turn_idx": ti,
                "num_prev_turns": ti,
                "situation": d["situation"],
                "emotion_type": d["emotion_type"],
                "problem_type": d["problem_type"],
                "context": [{"speaker": t["speaker"], "text": t["text"], "strategy": t.get("strategy")} for t in ctx],
                "query": last_usr,
                "response": turn["text"],
                "strategy": turn["strategy"],
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esconv", default="data/ESConv.json")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    data = [reformat(d) for d in json.load(open(args.esconv))]
    for mode in ("paper", "zcj"):
        os.makedirs(os.path.join(args.out, mode), exist_ok=True)
        for split, dialogs in split_dialogs(data, mode).items():
            ex = to_examples(dialogs, split)
            path = os.path.join(args.out, mode, f"{split}.jsonl")
            with open(path, "w") as f:
                for e in ex:
                    f.write(json.dumps(e) + "\n")
            print(f"{mode:5s} {split:5s} dialogs={len(dialogs):4d} examples={len(ex):5d} -> {path}")


if __name__ == "__main__":
    main()
