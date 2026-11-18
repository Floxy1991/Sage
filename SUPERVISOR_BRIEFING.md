# SAGE reproduction: briefing for the supervisor meeting

About a 10-minute read. Everything below was run on the school DGX.

## 1. What I reproduced, in one paragraph

The paper (Yang et al., *SAGE: Self-retrieval-augmented generative LLM for emotional support conversation*, ESWA 2026) proposes SAGE, a system for emotional-support chat. It picks a support strategy (e.g. "Question", "Reflection of feelings"), retrieves similar past supporter replies, and generates a response that uses them. The authors did **not** release code. So I **re-implemented** SAGE from the paper's description, trained it with the paper's settings (T5-base backbone, 100 epochs for the self-retrieval model), and compared it with the paper's numbers. I also ran the baselines the paper compares against: a BM25 retrieval floor and two open LLMs, Qwen2.5-7B and Mistral-7B. I evaluated everything on both of the paper's datasets, **ESConv** and **ESC-Eval**.

## 2. Where to see the results

| What | Where |
|---|---|
| All tables, ours next to the paper's | `sage-repro/results/dgx_100epoch/comparison.md` (project files). This is the file to show. |
| The same tables, regenerated live | Section 8 "Compare with the paper" in the notebook on the DGX |
| Raw outputs (predictions, per-dialogue scores) | On the DGX, in the notebook's `results/` folder: `sage_paper100/`, the ablation folders, `qwen2.5-7b-instruct/`, `mistral-7b-instruct-v0.3/`, `esceval_sage/`, `esceval_qwen2.5-7b-instruct/`, `esceval_mistral-7b-instruct-v0.3/` |
| Code | github.com/Floxy1991/Sage: `sage.py` (SAGE), `llm_baseline.py`, `bm25_baseline.py`, `esc_eval.py`, `metrics.py` |
| Paper problems and missing settings | `sage-repro/AUDIT.md` |

## 3. The key numbers

### ESConv (3,481 test examples, the paper's 7:1:2 split)

| System | Strategy ACC | B-2 | R-L | PPL |
|---|---|---|---|---|
| SAGE, paper | 46.80 | 11.87 | 20.51 | 14.32 |
| **SAGE, ours** | **22.92** | **6.32** | **15.54** | **16.60** |
| BM25 floor, ours | 21.57 | 4.45 | 12.76 | – |
| Qwen2.5-7B, ours (paper) | 13.59 (36.89) | 3.80 (10.89) | 11.24 (19.21) | 96.7 (14.58) |
| Mistral-7B, ours (paper) | 11.89 (38.91) | 3.91 (9.87) | 11.58 (18.92) | 34.9 (15.12) |

**How to say it:** "Our SAGE beats every baseline we ran on ESConv, as the paper claims, but it reaches about half the paper's absolute scores. It lands at roughly the level of the paper's older BlenderBot-Joint baseline (B-2 6.18, PPL 18.11)."

**Retrieval** (Recall@10): ours 36.1 vs paper 74.8. BM25 gets 11.4.

**Ablations** (paper Table 6): removing the constraint or the ResIDs did *not* hurt in our run; "w/o constraint" was even slightly better. Only retrieval-only generation was clearly worse. So the paper's claim that each component helps did not reproduce.

### ESC-Eval (331 role-play conversations per system, scored 1–5)

| System | Fluency | Empathy | Information | Skill | Overall |
|---|---|---|---|---|---|
| SAGE, paper | 4.12 | 4.15 | 3.94 | 4.08 | 4.12 |
| **SAGE, ours** | **2.43** | **3.65** | **1.87** | **1.88** | **1.55** |
| Qwen2.5-7B, ours | 3.63 | 3.95 | 2.93 | 2.88 | 1.58 |
| Mistral-7B, ours | 3.60 | 3.93 | 2.95 | 3.30 | 1.50 |

**How to say it:** "On ESC-Eval our SAGE does *not* beat the 7B LLMs, unlike in the paper. It is behind on fluency, information and skill, and about level on Overall."

## 4. Why our numbers differ from the paper's

Know these four points well. They are the core of the discussion.

1. **No code, many missing settings.** The paper omits many settings that matter: how the retrieval IDs are built for a query, the retrieval candidate pool, several hyperparameters, and the LLM prompt. I had to choose each of these, and every choice is marked *ASSUMPTION* in the code. AUDIT.md lists 12 missing settings.
2. **The paper has internal inconsistencies.** For example: its strategy distribution (Table A.9) doesn't match the public ESConv data; Tables 5 and C.15 report different numbers for the same thing; and it reports a "safety score" that it never defines. See AUDIT.md section 1 (13 items). This makes exact numbers hard to trust or match.
3. **ESC-Eval setup is different by necessity.** The paper uses 2,801 role cards that aren't public. I used the 331 English cards that ESC-Eval officially releases. The paper's Table 5 was scored by three of its own authors. I used ESC-Eval's official automatic judge (ESC-RANK) instead. ESC-RANK scores 0–4, so I added 1 to put it on the paper's 1–5 scale.
4. **Model size.** The paper says SAGE uses T5-base (about 220M parameters), and so does ours. In a 5-turn conversation, a small model like this gives shorter, more generic replies than a 7B chat model. That plausibly explains the ESC-Eval result. The paper doesn't explain how a T5-base model scores above 7B LLMs.

Also: the open LLMs did much worse for us than in the paper. We ran them zero-shot with the prompt the paper describes in words (it never prints the actual prompt). The paper's numbers likely come from a better-tuned prompt or setup that it doesn't report.

## 5. Questions the supervisor may ask

**"Did you just run their code?"**
No. There is no code. I re-implemented SAGE from the paper: a T5-base self-retrieval model, an RQ-VAE that gives each response a code ("ResID"), and a fusion generator. I trained them in that order on the DGX: 100 epochs for the self-retrieval model, 50 for the RQ-VAE, 3 for the fusion generator.

**"So did the reproduction fail?"**
Partly. The direction holds on ESConv: SAGE beats BM25 and the open LLMs. The paper's absolute numbers, its ablation claims and its ESC-Eval advantage did not reproduce. Given the missing settings and the inconsistencies in the paper, that is a legitimate reproduction finding, not just a failure on my side.

**"How do you know your metrics are computed correctly?"**
The metrics (BLEU, Distinct, ROUGE-L, accuracy, Recall@K, MRR, nDCG) are in `metrics.py`, following the paper's Section 4.2. The BM25 floor gives sensible, low numbers. Our SAGE lands close to a known published baseline (BlenderBot-Joint). Both suggest the pipeline behaves correctly.

**"Why is the LLM perplexity so high (Qwen 96.7)?"**
It is the perplexity of the *reference* reply under a zero-shot chat model. That is naturally high, because the model would phrase things differently. The paper doesn't say how it computed PPL for the LLMs. It even reports PPL for GPT-4o, which the API can't compute that way. So the PPL column isn't comparable.

**"Why is strategy accuracy the same for all ablations?"**
All ablations share the same trained self-retrieval model. The strategy is predicted by that model before the parts being removed (constraint, ResIDs) come into play.

**"Why are SAGE's and BM25's retrieval numbers not directly comparable?"**
SAGE ranks within a pool of 100 candidates, because the paper doesn't give a pool size. BM25 searches all 3,481 test responses.

**"What is ESC-Eval?"**
It is an evaluation in which a role-play model (ESC-Role, 14B) acts as a person with a problem, described by a "role card", and chats with the system for 5 turns. A judge model (ESC-RANK) then scores the conversation on 7 dimensions. It is from Zhao et al., EMNLP 2024, and the SAGE paper uses it as its second dataset.

**"Did anything need fixing to run it?"**
Yes. The ESC-RANK judge's own code crashed with the current version of the transformers library. The judge only outputs one digit (0–4), so I had it read off the most likely digit directly instead of generating text. It is the same judge and the same prompts.

**"What would you do next?"**
- Try the LLM baselines with a few-shot prompt, to see if they get closer to the paper.
- Look at sample SAGE dialogues from ESC-Eval, to see why fluency and information are low.
- Write to the authors for the missing settings ("data available on request").

## 6. Three sentences to open with

"I re-implemented SAGE, since no code was released, and trained it with the paper's settings on the DGX. I evaluated it on both of the paper's datasets against BM25, Qwen and Mistral. It beats those baselines on ESConv but reaches about half of the paper's reported scores, and on ESC-Eval it does not beat the LLMs; I've documented the missing settings and inconsistencies in the paper that explain much of the gap."
