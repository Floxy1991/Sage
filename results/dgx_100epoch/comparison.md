# SAGE, 100-epoch run on the school DGX (2026-11-11)

Full ESConv test set (paper 7:1:2 split, n = 3481). Self-retrieval LM trained for the paper's 100 epochs; other settings as in Tables B.11-B.12.

## Generation and strategy (Tables 1, 2, 6)

| System | Source | ACC | Macro-F1 | PPL | B-2 | B-4 | D-1 | D-2 | R-L |
|---|---|---|---|---|---|---|---|---|---|
| SAGE | paper | 46.80 | 38.95 | 14.32 | 11.87 | 3.88 | 9.24 | 39.86 | 20.51 |
| SAGE | ours | 22.92 | 19.82 | 16.60 | 6.32 | 2.14 | 3.97 | 19.63 | 15.54 |
| SAGE w/o constraint | ours | 22.92 | 19.82 | 16.56 | 6.62 | 2.30 | 4.34 | 22.25 | 15.55 |
| SAGE w/o ResID | ours | 22.92 | 19.82 | 16.62 | 6.47 | 2.09 | 3.87 | 18.78 | 15.66 |
| Retrieval only | ours | 22.92 | 19.82 | – | 4.27 | 0.79 | 2.95 | 17.99 | 13.52 |
| BM25 | ours | 21.57 | 18.67 | – | 4.45 | 0.92 | 4.55 | 27.93 | 12.76 |
| Qwen2.5-7B-Instruct | paper | 36.89 | 29.99 | 14.58 | 10.89 | 3.24 | 8.12 | 37.86 | 19.21 |
| Qwen2.5-7B-Instruct | ours | 13.59 | 9.14 | 96.7* | 3.80 | 0.66 | 3.23 | 21.05 | 11.24 |
| Mistral-7B-Instruct | paper | 38.91 | 31.39 | 15.12 | 9.87 | 2.94 | 6.85 | 35.14 | 18.92 |
| Mistral-7B-Instruct (v0.3) | ours | 11.89 | 5.35 | 34.86* | 3.91 | 0.72 | 2.56 | 15.15 | 11.58 |

Open LLMs (section 5) are zero-shot with the strategy prompt from Sec. 4.4, greedy decoding, full precision, run on the DGX on 2026-11-12. *PPL is the teacher-forced perplexity of the gold response given the prompt and gold strategy; 21 (Qwen) and 14 (Mistral) examples with a non-finite loss are skipped. The paper does not state its PPL setup for LLMs, so these PPL values are not directly comparable.

## Retrieval (Table 4)

| System | Source | R@1 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| SAGE | paper | 31.60 | 74.80 | 0.437 | 0.703 |
| SAGE | ours | 8.68 | 36.08 | 0.163 | 0.209 |
| BM25 | ours | 0.89 | 11.43 | 0.041 | 0.059 |

SAGE retrieval uses a candidate pool of 100 (the paper does not state its pool size); BM25 searches the full test pool of 3481, so the two retrieval rows are not directly comparable.

## Reading

- Our SAGE beats the BM25 floor on every generation metric except diversity, and is roughly level with the paper's BlenderBot-Joint baseline (B-2 6.18, PPL 18.11), but reaches about half of the paper's reported SAGE scores.
- Strategy accuracy is identical across all ablations, because they share the same self-retrieval model's strategy prediction.
- Removing the constraint or the ResIDs does not hurt; "w/o constraint" is slightly better. This does not reproduce the paper's Table 6, where each component helps. Only retrieval-only generation is clearly worse.
- The open LLMs score far below the paper's LLM rows (ACC 12–14 vs 37–39, B-2 about 3.9 vs 10–11) and below our own SAGE on every metric except Qwen's D-2. The paper's LLM numbers may come from a tuned prompt or few-shot setup it does not describe; ours are zero-shot with the prompt in llm_baseline.py.

## ESC-Eval (Table 5, Table C.15)

331 English high-quality ESC-Eval role cards, 5 turns, seeker ESC-Role, judge ESC-RANK (InternLM2-chat-7B + one LoRA per dimension), run on the DGX 2026-11-13 to 2026-11-16. ESC-RANK scores 0-4; shown here +1 to match the paper's 1-5 scale. Judge patch: one forward pass, most likely digit 0-4 (InternLM2's generate() crashes on current transformers). Paper rows: Table C.15 "Overall" (Flu./Exp./Emp./Info./Skill./Overall, automatic, 2,801 cards not public) and Table 5 (Hum., human raters).

| System | Source | Flu. | Exp. | Emp. | Info. | Hum. | Skill. | Overall |
|---|---|---|---|---|---|---|---|---|
| SAGE | paper | 4.12 | 4.28 | 4.15 | 3.94 | 4.28 | 4.08 | 4.12 |
| SAGE | ours | 2.43 | 2.70 | 3.65 | 1.87 | 2.19 | 1.88 | 1.55 |
| Qwen2.5-7B-Instruct | paper | 3.84 | 3.91 | 3.78 | 3.62 | 4.12 | 3.74 | 3.80 |
| Qwen2.5-7B-Instruct | ours | 3.63 | 3.32 | 3.95 | 2.93 | 2.00 | 2.88 | 1.58 |
| Mistral-7B-Instruct | paper (Table 5 only) | 4.18 | – | 3.62 | 3.71 | 3.95 | – | – |
| Mistral-7B-Instruct (v0.3) | ours | 3.60 | 3.47 | 3.93 | 2.95 | 1.99 | 3.30 | 1.50 |

n = 331 dialogues per system; every dialogue scored on all seven dimensions.

### Reading

- Unlike the paper, our SAGE does **not** beat the open LLMs on ESC-Eval. It is clearly behind on fluency (2.43 vs 3.6), exploration, information/suggestions (1.87 vs 2.9) and skill (1.88 vs 2.9–3.3), and slightly behind on empathy (3.65 vs 3.9). It is level on Overall (1.55 vs 1.50–1.58) and slightly ahead only on humanoid (2.19 vs 2.0).
- A plausible reason: our SAGE is a T5-base (220M) re-implementation trained only on ESConv, so its replies are short and generic next to 7B instruction-tuned models in a 5-turn conversation with a 14B role-play seeker. The paper also uses t5-base, and does not explain how it reaches its higher ESC-Eval scores (see AUDIT.md).
- All systems get low Overall and Hum. scores from ESC-RANK, so those two columns are compressed and say little about differences.
- Differences from the paper's setup: automatic judge for all columns (the paper's Table 5 is human-rated), 331 public cards instead of 2,801, 5 turns as in ESC-Eval's code, and the judge picks the most likely digit in one forward pass rather than generating text.
