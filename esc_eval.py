"""ESC-Eval, the second dataset in SAGE (Sec. 4.1, Table 5, Table C.15).

ESC-Eval (Zhao et al., EMNLP 2024) evaluates a support system by letting a
role-playing "seeker" model, driven by a role card, talk with it for several
turns, then scoring each dialogue on seven 0-4 dimensions. This follows the
official ESC-Eval code (evaluate.py / score.py):

  roleplay  ESC-Role (or any chat LLM) plays the card; the system under test
            answers. 5 turns, opening line and seeker prompt as in ESC-Eval.
  score     ESC-RANK (InternLM2-chat-7B + one LoRA per dimension) or, if that
            cannot be loaded, any chat LLM as judge with the same prompts.

Systems: "sage" (our re-implementation) or "llm:<hf model id>" (the open-LLM
baselines, prompted with the same strategy prompt as llm_baseline.py).

ASSUMPTIONS: SAGE reports a 5-point scale (Table 5) while ESC-Eval scores
0-4; we report raw 0-4 and the +1 shifted 1-5 values. SAGE says up to 20
turns (App. A.2.1); ESC-Eval's code uses 5, which we default to.

Usage:
  python esc_eval.py roleplay --system sage --out results/esceval_sage --limit 50
  python esc_eval.py roleplay --system llm:Qwen/Qwen2.5-7B-Instruct --out results/esceval_qwen --load-4bit
  python esc_eval.py score --dialogs results/esceval_sage/dialogs.json --judge escrank
"""
import argparse
import json
import os
import re
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CARDS = "data/esceval/card_high_en.json"
PROMPTS = "data/esceval/escrank_prompts_en.json"
SEEKER_PROMPT = ("I hope you can role-play as a person with worries and communicate with an AI assistant. "
                 "Here is your character card:'{card}\n' Please simulate a natural human conversation as much as "
                 "possible. Avoid behaving like a machine and refrain from mentioning that you are an AI assistant. "
                 "Gradually reveal your worries throughout our multiple conversations.")
OPENING = "Hello, I'm your personal assistant. You can confide in me about any worries or concerns you may have!"
DIMENSIONS = ["fluecy", "diversity", "empathy", "suggection effectiveness", "humanoid", "emotional knowledge",
              "human preference"]  # spelled as in ESC-Eval score.py, since ESC-RANK was trained on these prompts
KEYS = ["fluency", "diversity", "empathic", "suggestion", "human", "tech", "overall"]
# SAGE Table 5 / C.15 column -> ESC-Eval dimension.
SAGE_COLS = {"Flu.": "fluency", "Exp.": "diversity", "Emp.": "empathic", "Info.": "suggestion",
             "Hum.": "human", "Skill.": "tech", "Overall": "overall"}


class Chat:
    """Minimal greedy chat wrapper around a Hugging Face causal LM."""

    def __init__(self, model_id, load_4bit=False, max_new_tokens=256, adapters=None, trust_remote_code=False):
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16 if cuda else torch.float32
        kw = {"torch_dtype": dtype, "device_map": "auto" if cuda else None, "trust_remote_code": trust_remote_code}
        if load_4bit and cuda:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        if adapters:  # {name: path}, one LoRA per ESC-RANK dimension
            from peft import PeftModel
            names = list(adapters)
            self.model = PeftModel.from_pretrained(self.model, adapters[names[0]], adapter_name=names[0])
            for n in names[1:]:
                self.model.load_adapter(adapters[n], adapter_name=n)
        self.max_new_tokens = max_new_tokens

    def use(self, adapter):
        self.model.set_adapter(adapter)

    @torch.no_grad()
    def __call__(self, messages):
        try:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # templates without a system role (e.g. Mistral): fold it into the first user turn
            sys_msg = [m["content"] for m in messages if m["role"] == "system"]
            rest = [dict(m) for m in messages if m["role"] != "system"]
            if sys_msg and rest:
                rest[0]["content"] = sys_msg[0] + "\n\n" + rest[0]["content"]
            text = self.tok.apply_chat_template(rest, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def free(self):
        del self.model
        torch.cuda.empty_cache()


def _alternate(messages):
    """Chat templates such as Mistral's require user/assistant turns to alternate; the seeker
    sees the ESC-Eval opening as a user turn, so merge any consecutive same-role turns."""
    out = []
    for m in messages:
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n" + m["content"]}
        else:
            out.append(dict(m))
    return out


def make_system(spec, args):
    """Return reply(history) -> str, where history is [(speaker, text)] with speaker in {usr, sys}."""
    if spec == "sage":
        from sage import SageModel
        sage_args = SimpleNamespace(split=args.split, selfret=args.selfret, index=args.index, fusion=args.fusion,
                                    knowledge="none", eval_split="test", bs_strat=3, r=20, br=4, tau=1.0,
                                    delta=1.0, alpha=0.5, beta=0.5, no_constraint=False, no_resid=False,
                                    no_comet=False, no_heal=False, retrieval_only=False)
        model = SageModel(sage_args)

        def reply(history):
            ex = {"id": None, "context": [{"speaker": s, "text": t} for s, t in history],
                  "query": next(t for s, t in reversed(history) if s == "usr")}
            return model.respond(ex)["prediction"]
        return reply, None

    from llm_baseline import SYSTEM, parse
    chat = Chat(spec.split(":", 1)[1], load_4bit=args.load_4bit, max_new_tokens=128)

    def reply(history):
        convo = "\n".join(f"{'Help-seeker' if s == 'usr' else 'Supporter'}: {t}" for s, t in history[-10:])
        out = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Conversation so far:\n" + convo}])
        return parse(out)[1]
    return reply, chat


def roleplay(args):
    cards = json.load(open(args.cards))
    if args.limit:
        cards = cards[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "dialogs.json")
    dialogs = json.load(open(path)) if os.path.exists(path) else {}
    seeker = Chat(args.seeker, load_4bit=args.load_4bit, max_new_tokens=args.max_new_tokens)
    system, sys_chat = make_system(args.system, args)
    for i, card in enumerate(cards):
        key = str(card.get("id", i))
        if key in dialogs:
            continue
        seeker_msgs = [{"role": "system", "content": SEEKER_PROMPT.format(card=card["base"])},
                       {"role": "user", "content": OPENING}]
        history = [("sys", OPENING)]
        for _ in range(args.turns):
            said = seeker(_alternate(seeker_msgs))
            seeker_msgs.append({"role": "assistant", "content": said})
            history.append(("usr", said))
            ans = system(history) or "I see."
            seeker_msgs.append({"role": "user", "content": ans})
            history.append(("sys", ans))
        dialogs[key] = {"source": card.get("source"), "card": card["base"],
                        "dialog": [("AI assistant：" if s == "sys" else "ESC-Role：") + t for s, t in history]}
        json.dump(dialogs, open(path, "w"), ensure_ascii=False, indent=1)
        print(f"{i + 1}/{len(cards)}", flush=True)
    seeker.free()
    if sys_chat:
        sys_chat.free()
    return path


def load_judge(args):
    if args.judge == "escrank":
        from huggingface_hub import snapshot_download
        lora = snapshot_download(args.escrank_repo)
        adapters = {k: os.path.join(lora, f"{k}_en") for k in KEYS}
        return Chat(args.escrank_base, load_4bit=args.load_4bit, max_new_tokens=8, adapters=adapters,
                    trust_remote_code=True)
    return Chat(args.judge.split(":", 1)[1], load_4bit=args.load_4bit, max_new_tokens=8)


def score(args):
    dialogs = json.load(open(args.dialogs))
    prompts = json.load(open(PROMPTS))
    judge = load_judge(args)
    out_path = os.path.join(os.path.dirname(args.dialogs), f"scores_{args.judge.replace('/', '_').replace(':', '_')}.json")
    scores = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for n, (key, d) in enumerate(dialogs.items()):
        if key in scores:
            continue
        # Same formatting as ESC-Eval score.py (its role labels included).
        text = "\n\n".join(d["dialog"]).replace("AI assistant", "**AI助手**").replace("ESC-Role：", "**用户**")
        scores[key] = {}
        for p, dim, k in zip(prompts, DIMENSIONS, KEYS):
            if args.judge == "escrank":
                judge.use(k)
            msg = (p + "Dialogue between user and AI assistant: \n" + text.strip() + "\n"
                   + "Based on the rules, give your " + dim + " score (The number only) to the Dialogue.")
            m = re.search(r"[0-4]", judge([{"role": "user", "content": msg}]))
            scores[key][k] = int(m.group()) if m else -1
        json.dump(scores, open(out_path, "w"), indent=1)
        print(f"scored {n + 1}/{len(dialogs)}", flush=True)
    summary = summarise(scores)
    summary.update({"judge": args.judge, "dialogs": args.dialogs})
    json.dump(summary, open(os.path.join(os.path.dirname(args.dialogs), "esceval.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def summarise(scores):
    res = {"n": len(scores)}
    for k in KEYS:
        vals = [s[k] for s in scores.values() if s.get(k, -1) >= 0]
        res[k] = sum(vals) / len(vals) if vals else None
    # SAGE-style columns on a 1-5 scale (ESC-Eval 0-4 shifted by +1; ASSUMPTION, see module docstring).
    res["sage_scale_1to5"] = {c: (res[k] + 1 if res[k] is not None else None) for c, k in SAGE_COLS.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("roleplay")
    p.add_argument("--system", required=True, help='"sage" or "llm:<hf model id>"')
    p.add_argument("--seeker", default="haidequanbu/ESC-Role", help="role-playing model (ESC-Role is Qwen1.5-14B based)")
    p.add_argument("--cards", default=CARDS)
    p.add_argument("--out", required=True)
    p.add_argument("--turns", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--split", default="paper")
    p.add_argument("--selfret", default="ckpt/selfret")
    p.add_argument("--index", default="ckpt/index")
    p.add_argument("--fusion", default="ckpt/fusion")
    p = sub.add_parser("score")
    p.add_argument("--dialogs", required=True)
    p.add_argument("--judge", default="escrank", help='"escrank" or "llm:<hf model id>"')
    p.add_argument("--escrank-repo", default="haidequanbu/ESC-RANK")
    p.add_argument("--escrank-base", default="internlm/internlm2-chat-7b")
    p.add_argument("--load-4bit", action="store_true")
    a = ap.parse_args()
    roleplay(a) if a.cmd == "roleplay" else score(a)


if __name__ == "__main__":
    main()
