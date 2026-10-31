# Running SAGE on a laptop without an NVIDIA GPU

The kit now picks its device itself: NVIDIA CUDA if present, then Apple Silicon (MPS), then the CPU. Everything below works on a plain Windows laptop with an AMD or Intel GPU, on the CPU. The idea is to split the work in two:

- **On the laptop:** set-up, data checks, the BM25 baseline, a scaled-down SAGE run that exercises every stage end to end, a small-LLM baseline, inference with checkpoints trained elsewhere, and the comparison table.
- **On Colab (free T4):** the full-size training (t5-base on all 12,398 training examples), the 7B LLM baselines and ESC-Eval. These need a GPU or 30+ GB of RAM and would take days on a laptop CPU.

## What fits where

| Stage | Laptop CPU (16 GB RAM) | Needs Colab |
|---|---|---|
| BM25 baseline (step 0) | yes, ~15 min | |
| SAGE mini: t5-small, 3,000 train examples, 1 epoch, 50 test examples | yes, ~1-2 h | |
| SAGE full: t5-base, all data, 3 epochs | possible but ~2-4 days | yes, ~3-5 h on T4 |
| SAGE inference with Colab-trained checkpoints | yes, a few seconds per example; use `--limit` | full test set (3,481) is faster there |
| LLM baseline with Qwen2.5-0.5B or 1.5B | yes, 30-200 examples | |
| LLM baselines with the paper's 7B/8B models | no (28 GB RAM in fp32, ~1 min per reply) | yes |
| ESC-Eval role-play (14B seeker + 7B judge) | no | yes, needs 16 GB GPU |

Times come from a test run on a 4-core CPU with a model the size of t5-small (about 2.3 s per training step at batch size 8, about 4 s per test example at inference); a recent 8-core laptop on mains power should be similar or faster.

The mini runs prove the code works on your machine and let you read and change it comfortably. Their scores are **not** comparable with the paper, because the model is smaller, it sees a quarter of the data, and it trains for one epoch.

## 1. One-time set-up (Windows, PowerShell)

1. Install Python 3.11 or 3.12 from python.org (tick "Add python.exe to PATH") and Git from git-scm.com.
2. Get the code into your research folder:
   ```powershell
   cd "C:\Desktop\Research focus"
   git clone https://github.com/Floxy1991/Sage.git
   cd Sage
   ```
   (If you already have it: `git pull`.)
3. Make a virtual environment and install the CPU packages:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements-local.txt
   ```
   If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.
4. Check it:
   ```powershell
   python run_local.py check
   ```
   You should see `device: cpu (N threads) (works)` and `ok` next to the data files.

The first run of each stage downloads its model from Hugging Face into `%USERPROFILE%\.cache\huggingface` (t5-small 240 MB, t5-base 900 MB, Qwen2.5-0.5B 1 GB).

On a Mac, the same commands work in Terminal with `source .venv/bin/activate`; the check should print `device: mps`.

## 2. Run the pieces

**Prefer Jupyter?** Open `SAGE_reproduction.ipynb` (`pip install notebook`, then `jupyter notebook`) and run it top to bottom. With no NVIDIA GPU it switches to laptop mode by itself: t5-small, 3,000 training examples, 50 test examples, Qwen2.5-0.5B, and no ESC-Eval, about 2 hours in all. The commands below do the same thing as separate scripts.

Run each one on its own; every stage writes to `results/<name>/` and can be repeated.

```powershell
python run_local.py bm25          # BM25 floor, ~15 min
python run_local.py sage-mini     # SAGE end to end on t5-small, ~1-2 h
python run_local.py llm-mini      # Qwen2.5-0.5B on 30 test examples
python run_local.py compare       # table next to the paper's numbers
```

`sage-mini` runs the five SAGE stages one after another: train the self-retrieval LM, build the ResID index, train the fusion generator, run inference, evaluate retrieval. To make it shorter or longer:

```powershell
python run_local.py sage-mini --size tiny                 # 1,000 examples, 20 test, ~20-30 min: a smoke test
python run_local.py sage-mini --train-limit 6000 --epochs 2
python run_local.py sage-mini --size base                 # t5-base, the paper's backbone; ~3x slower
```

You can also run any stage by hand, exactly as in README.md, adding the laptop flags:

```powershell
python sage.py train-selfret --base t5-small --train-limit 3000 --epochs 1 --bs 8 --out ckpt/my/selfret
python sage.py build-index --model ckpt/my/selfret --train-limit 3000 --epochs 20 --out ckpt/my/index
python sage.py train-fusion --base t5-small --selfret ckpt/my/selfret --index ckpt/my/index --knowledge none --bs 8 --out ckpt/my/fusion
python sage.py infer --selfret ckpt/my/selfret --index ckpt/my/index --fusion ckpt/my/fusion --knowledge none --limit 50 --out results/my
```

New flags:

- `--device auto|cpu|mps|dml` on every `sage.py` and `llm_baseline.py` command (or set `SAGE_DEVICE`). `auto` is the default.
- `--train-limit N` on `train-selfret` and `build-index` trains on the first N training examples. `train-fusion`, `infer` and `eval-retrieval` follow whatever the index was built on, so you don't repeat it.
- `--load-4bit` is ignored without an NVIDIA GPU (it prints a note), so Colab commands still run.

## 3. Train on Colab, evaluate on the laptop

This gives real-size SAGE numbers without a laptop GPU.

1. Run sections 1 and 4 of `SAGE_colab.ipynb` (training only). The checkpoints land in `MyDrive/Sage/ckpt/` (selfret, index, fusion; about 2.7 GB).
2. Download that `ckpt` folder from Google Drive into `C:\Desktop\Research focus\Sage\ckpt`.
3. On the laptop:
   ```powershell
   python sage.py infer --knowledge none --limit 300 --out results/sage_local
   python sage.py eval-retrieval --limit 300 --out results/sage_local
   python run_local.py compare
   ```
   Checkpoints trained on a GPU load on the CPU without changes. Drop `--limit` for the full test set if you can leave it running overnight.

## AMD Radeon GPU (optional, experimental)

PyTorch has no CUDA for AMD integrated GPUs on Windows. Two options exist, and neither is needed:

- **DirectML:** `pip install torch-directml`, then add `--device dml`. It can speed up inference. Training T5 on it is often slower than the CPU and some operations fall back to the CPU, so keep training on `cpu`.
- **ROCm** works only on Linux and only for some Radeon cards; integrated Radeon 800M-series GPUs are not officially supported.

## Tips

- Close other heavy apps. t5-small with `--bs 8` needs a few GB of RAM; t5-base needs roughly three times as much. If you run out of memory, use `--bs 4`.
- Plug the laptop in and set Windows power mode to "Best performance"; on battery the CPU throttles and runs take 2-3x longer.
- If a run is cut off, re-run the same command. `llm_baseline.py` resumes where it stopped; a SAGE stage starts that stage over, but earlier stages are kept.
- `UnicodeDecodeError` on Windows means an old copy of the code: `git pull`. Files are now read as UTF-8 everywhere.
