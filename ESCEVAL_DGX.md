# ESC-Eval (paper section 7) on the DGX

Section 5 (Qwen2.5-7B and Mistral-7B on ESConv) is already finished on the DGX. Its numbers are in
`results/dgx_100epoch/comparison.md`. What remains is **ESC-Eval**: a role-play "seeker" model
(ESC-Role, 14B) talks to each system for 5 turns on each of the 331 English role cards, and then a judge
(ESC-RANK, InternLM2-7B) scores every dialogue on 7 dimensions. It runs for three systems: our SAGE,
Qwen2.5-7B and Mistral-7B.

Work in the same notebook you used for the 100-epoch run (`SAGE_100epochs.ipynb`), in JupyterLab on the DGX.

## Step 1. Resume (2 minutes)

Run the **Resume** cell near the top. It reloads the settings, the data and the section 7 functions.
Then run `%pip install einops` in a new cell once: the ESC-RANK judge (InternLM2) needs it.

## Step 2. Quick check on 2 role cards (about 15–30 minutes, most of it downloading the models)

Add a new cell at the end of the notebook, paste this in and run it. It checks that the role-play model
and the judge both load, and it estimates how long the full run will take. If the official judge
(ESC-RANK) won't load, it switches to a backup judge on its own and remembers that for step 3.

```python
import json, os, time, traceback
from types import SimpleNamespace

LOAD_4BIT = False                      # the DGX has enough memory for full precision, as in section 5
SEEKER = "haidequanbu/ESC-Role"
BACKUP_JUDGE = "llm:Qwen/Qwen2.5-14B-Instruct"

def play(system, out, limit):
    return roleplay(SimpleNamespace(system=system, seeker=SEEKER, cards=CARDS, out=out, turns=5, limit=limit,
                                    max_new_tokens=256, load_4bit=LOAD_4BIT, split=SPLIT,
                                    selfret="ckpt/selfret", index="ckpt/index", fusion="ckpt/fusion"))

def judge(path, which):
    return score(SimpleNamespace(dialogs=path, judge=which, escrank_repo="haidequanbu/ESC-RANK",
                                 escrank_base="internlm/internlm2-chat-7b", load_4bit=LOAD_4BIT))

t0 = time.time()
path = play("sage", "results/esceval_check", 2)
free_memory()
t_play = time.time() - t0
print(f"role-play: {t_play / 60:.1f} min for 2 cards (includes loading the models)")

t0 = time.time()
try:
    judge(path, "escrank")
    JUDGE = "escrank"
except Exception:
    traceback.print_exc()
    print("\n>>> ESC-RANK did not load. Using the backup judge", BACKUP_JUDGE)
    free_memory()
    judge(path, BACKUP_JUDGE)
    JUDGE = BACKUP_JUDGE
free_memory()
open("results/esceval_judge.txt", "w").write(JUDGE)
t_score = time.time() - t0
print(f"\nJUDGE = {JUDGE}")
print(f"Rough estimate for the full run: {3 * 331 / 2 * (t_play + t_score) / 3600:.0f} hours or less "
      "(the estimate counts model loading many times over, so the real run is usually shorter)")
print(json.dumps(json.load(open(path)), indent=1, ensure_ascii=False)[:3000])
```

**Send me a screenshot of the end of the output.** It shows which judge is in use, the time estimate
and the first dialogue, so I can check the role-play looks sensible before you spend many hours on it.

## Step 2b. Fix for the ESC-RANK judge (run once after every Resume)

InternLM2's own code (the base of ESC-RANK) crashes inside `generate()` on new versions of transformers
("causal mask" size mismatch). The judge only needs to output one digit, 0–4, so this cell skips `generate()`.
It does one plain forward pass without a cache and picks the most likely digit. Run it before step 3.
Because it redefines a function, run it again whenever you have run Resume.

```python
import torch

class DigitJudge:
    """ESC-RANK judge without generate(): one forward pass, most likely of the digits 0-4."""
    def __init__(self, chat):
        self.chat, tok = chat, chat.tok
        self.ids = {}
        for d in "01234":
            cand = set(tok.encode(d, add_special_tokens=False)[-1:])
            for t in (d, "▁" + d):
                i = tok.convert_tokens_to_ids(t)
                if isinstance(i, int) and i != tok.unk_token_id:
                    cand.add(i)
            self.ids[d] = sorted(cand)
        print("digit token ids:", self.ids)

    def use(self, adapter):
        self.chat.use(adapter)

    def free(self):
        self.chat.free()

    @torch.no_grad()
    def __call__(self, messages):
        tok, model = self.chat.tok, self.chat.model
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits[0, -1].float()
        return max(self.ids, key=lambda d: max(logits[i].item() for i in self.ids[d]))

if "_load_judge_orig" not in globals():
    _load_judge_orig = load_judge
def load_judge(args):
    j = _load_judge_orig(args)
    return DigitJudge(j) if args.judge == "escrank" else j
print("ESC-RANK digit judge patch active")
```

## Step 3. Full run (many hours; leave it running)

Add another cell and run it. It does all 331 cards for SAGE, then Qwen, then Mistral, and scores each one.

```python
import os, time
from types import SimpleNamespace

LOAD_4BIT = False
SEEKER = "haidequanbu/ESC-Role"
JUDGE = open("results/esceval_judge.txt").read().strip() if os.path.exists("results/esceval_judge.txt") else "escrank"
print("judge:", JUDGE)

for system in ["sage", "llm:Qwen/Qwen2.5-7B-Instruct", "llm:mistralai/Mistral-7B-Instruct-v0.3"]:
    name = "sage" if system == "sage" else system.split("/")[-1].lower()
    t0 = time.time()
    path = roleplay(SimpleNamespace(system=system, seeker=SEEKER, cards=CARDS, out=f"results/esceval_{name}",
                                    turns=5, limit=331, max_new_tokens=256, load_4bit=LOAD_4BIT, split=SPLIT,
                                    selfret="ckpt/selfret", index="ckpt/index", fusion="ckpt/fusion"))
    free_memory()
    score(SimpleNamespace(dialogs=path, judge=JUDGE, escrank_repo="haidequanbu/ESC-RANK",
                          escrank_base="internlm/internlm2-chat-7b", load_4bit=LOAD_4BIT))
    free_memory()
    print(f"=== {name} finished in {(time.time() - t0) / 3600:.1f} h ===", flush=True)
print("ESC-EVAL ALL DONE")
```

- You can close the browser and the virtual desktop. The cell keeps running on the DGX.
- When you come back, the cell shows `scored N/331` or `N/331` as its last lines while it is still
  working, and `ESC-EVAL ALL DONE` when it has finished.
- **If it stops** (an error, or the server restarted), run the Resume cell, then this cell again. It
  carries on from the last finished dialogue, because every dialogue and score is saved as it goes.
- **Out of memory:** someone else is using the GPU. Wait, or set `LOAD_4BIT = True` in the cell and run it again.

## Step 4. Results

When it says `ESC-EVAL ALL DONE`, run section **8 (Compare with the paper)**. It now includes the ESC-Eval
table. Send me a screenshot of it and I'll add it to the comparison.

## What to expect

- Our scores use the paper's **automatic** scorer on 331 public cards. The paper's Table 5 is from human
  raters, and its Table C.15 averages 2,801 cards that aren't public. So the numbers will not match
  exactly; what matters is whether SAGE ranks above Qwen and Mistral.
- ESC-RANK scores 0–4. The comparison shows these shifted to the paper's 1–5 scale (+1).
- If the backup judge had to be used, the scores are from Qwen2.5-14B rather than ESC-RANK. That is fine
  for comparing our three systems with each other, but it is not the paper's scorer, and the write-up
  should say so.
