# SAGE reproducibility audit

Paper: Yang, Hong, Jo, Oh. *SAGE: Self-retrieval-augmented generative LLM for emotional support conversation*. Expert Systems with Applications 313 (2026) 131524.
Checked against: the uploaded PDF and the public ESConv.json (thu-coai, 1,300 dialogs).

## Bottom line

SAGE can only be **re-implemented**, not rerun. There is no code ("data will be made available on request"), and many settings that decide the headline numbers are not reported. Several reported statistics also don't match the public data or each other. A faithful comparison is possible for the **baselines** (the open LLMs, BlenderBot-Joint). The SAGE numbers can only be compared as "our re-implementation under stated assumptions".

## 1. Checked facts that do not hold

| # | Paper says | What I found |
|---|---|---|
| 1 | Table A.9: 38,357 strategy-labelled utterances, with Restatement 0.5%, Information 1.2%, Self-disclosure 1.5% and Suggestions 3.8% | ESConv has strategy labels only on the **18,376 supporter** turns: Question 20.7%, Others 18.2%, Suggestions 16.1%, Affirmation 15.4%, Self-disclosure 9.3%, Reflection 7.8%, Information 6.6%, Restatement 5.9%. These match the official README. The paper's table can't come from this dataset. |
| 2 | Section 4.1: "standard split" 910 / 130 / 260 (7:1:2) | The widely used ESConv preprocessing (`codes_zcj/_reformat/process.py`, seed 13) splits 70/15/15. Baseline numbers taken from earlier papers may be on a different split. No seed or dialog list is given for 7:1:2. |
| 3 | Section 4.2.1: Table 1 "ACC" is Macro-F1 | Table 2 lists SAGE Top-1 ACC 46.80 and Macro-F1 38.95, and Table 1's ACC equals Table 2's Top-1 ACC. So Table 1 reports accuracy, not Macro-F1. |
| 4 | Section 5: SAGE beats KEMI "without relying on external knowledge bases (HEAL)" | SAGE uses HEAL in re-ranking (Eq. 5) and generation (Eq. 6), and it has a "w/o HEAL" ablation. |
| 5 | Section 6.1: removing HEAL lowers preference by 2.9% | Table 6: 52.7 to 49.8 is a 5.5% drop. |
| 6 | Table 5 (human eval): SAGE Emp 4.12, Flu 4.39; GPT-4o Emp 3.76, Flu 4.31 | Table C.15 "Overall": SAGE Flu 4.12, Emp 4.15; GPT-4o Flu 3.76. The columns look shifted between the two tables. |
| 7 | Table C.15: "average across 2801 dialogues" | Section 5.5 says three authors did the human evaluation. Rating 2,801 multi-turn dialogues for 6 systems is implausible, and the authors weren't blind to which system was theirs. |
| 8 | Section 5.4 text: long contexts are "≥5 turns" | Table 8 buckets are 1-2, 3-4 and ≥7. There is no 5-6 bucket. |
| 9 | Table 1: GPT-4o PPL 14.82 | The GPT-4o API can't score reference text with teacher forcing, so the paper doesn't say how this was computed. |
| 10 | Section 6.1: a "safety score" drops 7.6% without constraints | Safety is never defined or reported in any table. |

The arithmetic on the headline relative gains (13.4%, 52.1%, 68.0%, 20.7%, 8.6%, ablation deltas other than #5) checks out.

| 11 | Section 4.1, Table A.10: ESC-Eval has 2,801 role cards drawn from ESConv (1,300), EmpatheticDialogues, BlendedSkillTalk, Wizard of Wikipedia, PersonaChat and Topical-Chat | ESC-Eval's official release ([haidequanbu/ESC-Eval](https://github.com/haidequanbu/ESC-Eval)) has 655 high-quality cards: 331 English (ESConv 158, MHP 73, ExTES 70, Psych 25, EPITOME 5) and 324 Chinese (PsyQA 252, smile 72). The source mix in Table A.10 doesn't match the ESC-Eval paper or repo, and the 2,801-card pool isn't public. |
| 12 | Table 5: ESC-Eval scores on a "5-point scale"; App. A.2.1: up to 20 turns and EmoLlama-Chat-7B satisfaction | ESC-Eval rates 0-4 on seven dimensions over 5 turns, and scores with its ESC-RANK model. SAGE reports no EmoLlama numbers. |
| 13 | ESC-Eval uses ESConv-derived role cards | 158 of the 331 English cards come from ESConv dialogues. If some come from SAGE's ESConv training split, the ESC-Eval comparison leaks training data. The paper doesn't address this. |

## 2. Missing settings a re-implementation must guess

1. **Backbone.** The paper calls it "a single LLM", but Appendix B says the base model is **T5-base**. An unnamed "SFT" model is also trained (max length 2048, cosine schedule, 3 epochs).
2. **RQ-VAE.** Appendix B.2.1 describes 4 levels × 512 codes, but Table B.12 says "Code length: 1". It doesn't say what the input embedding is (which encoder, and whether it reconstructs text or vectors).
3. **Query ResID.** ResIDs are built from responses. The paper doesn't say how the query u_t gets a ResID for the Hamming-distance lookup.
4. **Retrieval evaluation.** It doesn't define what a "correct response" is, how big the candidate pool is, or how the DPR baseline was set up. This decides every number in Tables 4 and 8.
5. **Strategy classifier.** The architecture and inputs (is the strategy history included?) aren't given.
6. **Hyperparameters missing:** B_s, R, B_r, τ, δ, α, β, K (HEAL top-K) and λ1-λ3.
7. **Self-evaluation P_reject.** The paper doesn't say how the model learns to reject candidates.
8. **Fusion generator (Eq. 6).** The encoders, the decoder and its training data aren't described. Final decoding mixes beam search with nucleus sampling.
9. **COMET.** The version isn't given (ATOMIC-2020 BART is assumed), nor the relations (xIntent, xNeed, xEffect, xWant are assumed), nor the embedding model used for cosine similarity.
10. **HEAL.** The paper doesn't say how expert responses are indexed and retrieved, or which embedding is used.
11. **LLM baselines.** The prompt template isn't printed, and sampling settings for GPT-4o and the open models aren't given.
12. **Runs.** Results are said to be "averages of three runs", but seeds and standard deviations aren't reported.

## 3. Reproduction plan

| Stage | What | Compute | Compare to |
|---|---|---|---|
| 0 | Data prep for both splits (paper 7:1:2 with a fixed seed, and the common 70/15/15 seed-13 split), plus a metrics harness (ACC, Macro-F1, B-2/4, D-1/2, ROUGE-L, R@K, MRR, nDCG, entropy, JSD bias) | CPU, done here | n/a |
| 1 | Open-LLM baselines: Qwen-2.5-7B, LLaMA-3.1-8B and Mistral-7B, using the paper's described prompt (context + 8 strategy descriptions, pick a strategy, then respond) | 1 GPU with ≥16 GB, or 8 GB in 4-bit | Table 1 LLM rows, Tables 2-3 |
| 2 | BlenderBot-Joint with the official `codes_zcj` code | 1 GPU with ≥12 GB | Table 1 row 1 |
| 3 | SAGE re-implementation on T5-base with the assumptions above, plus the ablations from Table 6 | 1 GPU with ≥16 GB, several hours | Tables 1, 2, 4, 6 |
| (opt.) | GPT-4o baseline | OpenAI API key | Table 1 |

ESC-Eval (the second dataset) is reproduced with its official public cards, ESC-Role as the seeker and ESC-RANK as the judge (`esc_eval.py`, notebook section 7). SAGE's own Table 5 comes from three of its authors rating responses, which can't be repeated, so that comparison is indicative only.
