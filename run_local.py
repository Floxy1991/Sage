"""Run the reproduction in small pieces on a laptop without an NVIDIA GPU.

Each stage is independent and can be re-run on its own. Results land in results/<name>/,
the same place the Colab notebook writes them, so compare.py reads both.

  python run_local.py check          # 1 min: Python, PyTorch, device, data
  python run_local.py bm25           # ~15 min CPU: BM25 retrieval baseline (no model download)
  python run_local.py sage-mini      # ~1-2 h CPU: whole SAGE pipeline on t5-small and a data subset
  python run_local.py llm-mini       # ~20-40 min CPU: Qwen2.5-0.5B baseline on 30 test examples
  python run_local.py compare        # print our numbers next to the paper's
  python run_local.py all            # all of the above in order

sage-mini options: --size tiny|small|base (default small), --train-limit, --test-limit, --epochs.
Pass --device cpu|mps|dml to force a device (default: auto; see device.py).

The mini runs check that everything works end to end on your machine. Their scores are NOT
comparable with the paper (smaller model, less data, fewer epochs). Full-size runs go on Colab
(SAGE_colab.ipynb); you can then copy the ckpt/ folder back and run `infer` locally (LOCAL.md).
"""
import argparse
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

SIZES = {  # base model, default train subset, default test subset
    "tiny": ("t5-small", 1000, 20),
    "small": ("t5-small", 3000, 50),
    "base": ("t5-base", 3000, 50),
}


def run(args, label):
    print(f"\n=== {label}\n$ python {' '.join(args)}", flush=True)
    t = time.time()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY] + args, cwd=HERE, env=env)
    print(f"=== {label}: {'done' if r.returncode == 0 else 'FAILED'} in {(time.time() - t) / 60:.1f} min", flush=True)
    if r.returncode:
        sys.exit(r.returncode)


def check(a):
    print("Python", sys.version.split()[0], "on", platform.platform())
    try:
        import torch
        import transformers
    except ImportError as e:
        sys.exit(f"Missing package ({e}). Run: pip install -r requirements-local.txt")
    import device
    dev = device.pick(a.device)
    x = torch.randn(256, 256, device=dev)
    _ = (x @ x).sum().item()
    print("torch", torch.__version__, "| transformers", transformers.__version__)
    print("device:", device.describe(dev), "(works)")
    try:
        import psutil
        print(f"RAM: {psutil.virtual_memory().total / 2 ** 30:.0f} GB")
    except ImportError:
        pass
    for p in ("data/paper/train.jsonl", "data/paper/test.jsonl", "data/esceval/card_high_en.json"):
        print(("ok      " if os.path.exists(os.path.join(HERE, p)) else "MISSING ") + p)


def bm25(a):
    run(["bm25_baseline.py", "--split", "paper", "--out", "results/bm25_paper"], "BM25 baseline")


def sage_mini(a):
    base, tl, te = SIZES[a.size]
    base = a.base or base
    tl = a.train_limit or tl
    te = a.test_limit or te
    tag = f"local_{a.size}"
    ck = f"ckpt/{tag}"
    dev = ["--device", a.device]
    run(["sage.py", "train-selfret", "--base", base, "--epochs", str(a.epochs), "--bs", "8",
         "--train-limit", str(tl), "--out", f"{ck}/selfret"] + dev, "1/5 self-retrieval LM")
    run(["sage.py", "build-index", "--model", f"{ck}/selfret", "--epochs", "20",
         "--train-limit", str(tl), "--out", f"{ck}/index"] + dev, "2/5 ResID index (RQ-VAE)")
    run(["sage.py", "train-fusion", "--base", base, "--epochs", str(a.epochs), "--bs", "8",
         "--selfret", f"{ck}/selfret", "--index", f"{ck}/index", "--knowledge", "none",
         "--out", f"{ck}/fusion"] + dev, "3/5 fusion generator")
    # Smaller search than the paper (R=20, B_r=4) to keep CPU inference to a few seconds per example.
    run(["sage.py", "infer", "--selfret", f"{ck}/selfret", "--index", f"{ck}/index", "--fusion", f"{ck}/fusion",
         "--knowledge", "none", "--limit", str(te), "--r", "10", "--br", "2",
         "--out", f"results/sage_{tag}"] + dev, "4/5 SAGE inference")
    run(["sage.py", "eval-retrieval", "--selfret", f"{ck}/selfret", "--index", f"{ck}/index",
         "--limit", str(te * 2), "--out", f"results/sage_{tag}"] + dev, "5/5 retrieval eval")


def llm_mini(a):
    run(["llm_baseline.py", "--model", a.llm, "--split", "paper", "--limit", "30", "--batch-size", "2",
         "--out", "results/llm_local_" + a.llm.split("/")[-1].lower(), "--device", a.device], "small LLM baseline")


def compare(a):
    run(["compare.py"], "compare with paper")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "bm25", "sage-mini", "llm-mini", "compare", "all"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--size", choices=list(SIZES), default="small")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--base", default="", help="override the T5 checkpoint (hub id or local folder)")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
    a = ap.parse_args()
    stages = {"check": check, "bm25": bm25, "sage-mini": sage_mini, "llm-mini": llm_mini, "compare": compare}
    for s in (list(stages) if a.stage == "all" else [a.stage]):
        stages[s](a)


if __name__ == "__main__":
    main()
