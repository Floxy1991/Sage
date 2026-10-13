# SAGE reproduction kit

Reproduces *SAGE: Self-retrieval-augmented generative LLM for emotional support conversation* (Yang et al., ESWA 2026, [doi:10.1016/j.eswa.2026.131524](https://doi.org/10.1016/j.eswa.2026.131524)) on ESConv. The paper is included as [SAGE_paper.pdf](SAGE_paper.pdf) (CC BY-NC-ND 4.0). ESConv ([thu-coai](https://github.com/thu-coai/Emotional-Support-Conversation)) is included for academic research use only. The authors released no code, so the SAGE part is a **re-implementation**. See [AUDIT.md](AUDIT.md) for what the paper leaves out or gets wrong.

## Quick start on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Floxy1991/Sage/blob/main/SAGE_colab.ipynb) Runs every step on a free T4 GPU and saves outputs to Google Drive. `python compare.py` prints our numbers next to the paper's.

## Setup (on a machine with an NVIDIA GPU)

```bash
pip install torch transformers accelerate sentencepiece nltk scikit-learn rank-bm25 bitsandbytes
python prepare_data.py --esconv data/ESConv.json --out data   # already done: data/paper and data/zcj are committed
```

## Runs, in order

| Step | Command | Compare to | GPU / time (rough) |
|---|---|---|---|
| 0 | `python bm25_baseline.py --split paper --out results/bm25_paper` | sanity floor | CPU, 15 min (result in results/bm25_paper) |
| 1 | `python llm_baseline.py --model Qwen/Qwen2.5-7B-Instruct --out results/qwen` | Table 1-3, Qwen row | 16 GB, or 8 GB with `--load-4bit` |
| 1 | same with `meta-llama/Llama-3.1-8B-Instruct` (gated, needs HF login) and `mistralai/Mistral-7B-Instruct-v0.3` | LLaMA, Mistral rows | same |
| 2 | `python sage.py train-selfret` | n/a | 12 GB |
| 3 | `python sage.py build-index` | n/a | small |
| 4 | `python sage.py knowledge --comet <COMET-ATOMIC-2020 BART dir> --heal <heal.jsonl>` (optional) | n/a | small |
| 5 | `python sage.py train-fusion` | n/a | 12 GB |
| 6 | `python sage.py infer --out results/sage` | Table 1-3, SAGE row | 8 GB |
| 7 | `python sage.py eval-retrieval --out results/sage` | Table 4 | 8 GB |
| 8 | Ablations: `infer --no-constraint`, `--no-resid`, `--no-comet`, `--no-heal`, `--retrieval-only`; and `train-selfret --no-strategy-token` then infer | Table 6 | as 6 |

Add `--limit 200` to any `infer` or `llm_baseline` run for a quick check first. Use `--split zcj` to rerun on the 70/15/15 split that the earlier baselines used.

## Assumptions made where the paper is silent

- Backbone is T5-base (Appendix B), trained for 3 epochs at lr 1e-4 with batch size 16. One model does both generation, "context -> [STRAT] s [RESP] r", and self-evaluation, "judge: context response -> yes/no", with 1:1 random negatives drawn from other dialogues. P_valid is P(yes).
- ResIDs come from a 4 × 512 RQ-VAE (MLP encoder/decoder, 256-d codes, Sinkhorn ε 0.1, 10 iterations, 50 epochs) over mean-pooled T5 encoder vectors of each training response. The query gets its ResID by passing the context vector through the same quantiser. Ranking is by Hamming distance, with cosine breaking ties.
- Inference settings: B_s = 3 strategies, R = 20 candidates per strategy, B_r = 4 beams, τ = δ = 1, α = β = 0.5 (tune on valid with `--eval-split valid`).
- COMET infers xIntent, xNeed, xEffect and xWant from the last seeker utterance. HEAL supplies the top 3 expert responses by embedding similarity. Both are optional. Without them, Eq. 5 and Eq. 6 reduce to the context-only terms.
- The fusion generator is a separate T5-base. Its input is the strategy, the retrieved candidate and the context, cross-attended to COMET and HEAL as in Eq. 6. λ is learned and softmax-normalised. Decoding uses beam 4, repetition penalty 1.2, 3-gram blocking and min length 10.
- PPL is the gold response's perplexity under the generator, given the gold strategy.
- The retrieval pool for Table 4 is 100 test responses including the gold one. The paper doesn't state its pool size. On the full test pool, BM25 gets R@10 11.4, compared with 61.8 for the paper's DPR.
- Metrics: corpus BLEU (nltk, smoothing 1), corpus Distinct, ROUGE-L with β = 1.2, entropy and JSD in log2.

## Not reproducible as described
Human evaluation (Table 5) and ESC-Eval role-play (Table C.15) were done by three of the authors, and the protocol isn't released. GPT-4o's PPL can't be computed through the API.
