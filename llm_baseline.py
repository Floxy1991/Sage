"""Open-LLM baselines from SAGE Table 1-3 (Qwen-2.5-7B, LLaMA-3.1-8B, Mistral-7B).

Follows Sec. 4.4: "Each model is provided with a dialogue context along with
descriptions of eight support strategies and is instructed to select an
appropriate strategy and generate a response according to that strategy."
The paper does not print its template or decoding settings; ours are below
and are greedy by default.

PPL is the teacher-forced perplexity of the gold response, conditioned on the
prompt plus the gold strategy line.

Usage (GPU):
  python llm_baseline.py --model Qwen/Qwen2.5-7B-Instruct --split paper --out results/qwen
  add --load-4bit on GPUs with less than 16 GB; --limit 200 for a quick check.
"""
import argparse
import json
import math
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import generation_metrics, strategy_metrics
from prepare_data import STRATEGIES

STRATEGY_DESC = {
    "Question": "Ask for information related to the problem to help the help-seeker articulate their issues.",
    "Restatement or Paraphrasing": "A simple, more concise rephrasing of the help-seeker's statements.",
    "Reflection of feelings": "Articulate and describe the help-seeker's feelings.",
    "Self-disclosure": "Divulge similar experiences you have had or emotions you share, to express empathy.",
    "Affirmation and Reassurance": "Affirm the help-seeker's strengths, motivation, and capabilities, and provide reassurance and encouragement.",
    "Providing Suggestions": "Provide suggestions about how to change, but be careful not to overstep and tell them what to do.",
    "Information": "Provide useful information to the help-seeker, for example with data, facts, opinions, resources, or by answering questions.",
    "Others": "Exchange pleasantries and use other support strategies that do not fall into the above categories.",
}

SYSTEM = (
    "You are a supporter in an emotional support conversation. Choose the single most appropriate "
    "support strategy for your next reply from the list below, then write the reply using that strategy.\n\n"
    + "\n".join(f"- {s}: {d}" for s, d in STRATEGY_DESC.items())
    + "\n\nAnswer in exactly this format:\nStrategy: <one strategy name from the list>\nResponse: <your reply, one or two sentences>"
)


def user_prompt(ex, max_turns=10):
    lines = [f"Situation: {ex['situation']}", "", "Conversation so far:"]
    for t in ex["context"][-max_turns:]:
        lines.append(f"{'Help-seeker' if t['speaker'] == 'usr' else 'Supporter'}: {t['text']}")
    return "\n".join(lines)


def build_prompt(tok, ex):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(ex)}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:  # Mistral templates reject a system role
        msgs = [{"role": "user", "content": SYSTEM + "\n\n" + user_prompt(ex)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse(text):
    m = re.search(r"Strategy:\s*(.+)", text)
    raw = m.group(1).strip().lower() if m else ""
    strat = next((s for s in STRATEGIES if s.lower() in raw or raw.startswith(s.split()[0].lower())), "Others")
    r = re.search(r"Response:\s*(.+)", text, re.S)
    resp = (r.group(1) if r else text).strip().split("\n")[0]
    return strat, resp


@torch.no_grad()
def gold_nll(model, tok, prompt, ex):
    target = f"Strategy: {ex['strategy']}\nResponse: "
    full = prompt + target + ex["response"]
    pre = tok(prompt + target, return_tensors="pt", add_special_tokens=False).input_ids
    ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    labels = ids.clone()
    labels[:, : pre.shape[1]] = -100
    loss = model(input_ids=ids, labels=labels).loss
    n = int((labels != -100).sum())
    return float(loss) * n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="paper")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-ppl", action="store_true")
    args = ap.parse_args()

    test = [json.loads(l) for l in open(f"data/{args.split}/test.jsonl")]
    if args.limit:
        test = test[: args.limit]
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    pred_path = os.path.join(args.out, "predictions.jsonl")
    done = {json.loads(l)["id"] for l in open(pred_path)} if os.path.exists(pred_path) else set()
    todo = [e for e in test if e["id"] not in done]
    with open(pred_path, "a") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            prompts = [build_prompt(tok, e) for e in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for e, p, t in zip(batch, prompts, texts):
                strat, resp = parse(t)
                row = {"id": e["id"], "response": e["response"], "strategy": e["strategy"],
                       "prediction": resp, "pred_strategy": strat, "raw": t}
                if not args.no_ppl:
                    row["nll"], row["ntok"] = gold_nll(model, tok, p, e)
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)

    rows = [json.loads(l) for l in open(pred_path)]
    res = {"model": args.model, "split": args.split, "n": len(rows)}
    res.update(generation_metrics([r["response"] for r in rows], [r["prediction"] for r in rows]))
    res.update(strategy_metrics([r["strategy"] for r in rows], [r["pred_strategy"] for r in rows]))
    if all("nll" in r for r in rows):
        res["PPL"] = math.exp(sum(r["nll"] for r in rows) / sum(r["ntok"] for r in rows))
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
