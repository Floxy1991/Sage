# Interpreting the DGX 100-epoch results

A plain-language reading of `comparison.md`, written for the supervisor meeting.

## 1. The headline
We rebuilt SAGE from the paper's description alone, because the authors released no code. The paper's *ranking* on ESConv holds up: our SAGE beats every baseline we ran. The *absolute numbers* don't: we get about half of what the paper reports. Two of the paper's claims did not reproduce at all. The components don't help in the ablation study, and SAGE doesn't beat the LLMs on ESC-Eval.

## 2. ESConv (3,481 test examples)
**Strategy prediction (ACC / Macro-F1).** Ours scores 22.9 / 19.8 and the paper reports 46.8 / 39.0. There are 8 strategies, so random guessing gets about 12.5%. Our model learned something, but not much: it is only about 1.4 points above BM25 (21.6), which just copies the strategy of the most similar past reply.

**Response quality (B-2, B-4, R-L).** BLEU and ROUGE measure word overlap with the real supporter's reply. Ours gets B-2 6.32 against the paper's 11.87, and R-L 15.5 against 20.5. That is better than BM25 (4.45 / 12.8), so generating a reply beats copying a retrieved one. It is about the level of the paper's older BlenderBot-Joint baseline (B-2 6.18).

**Perplexity (PPL, lower is better).** Ours is 16.6 and the paper's is 14.3. This is the closest metric to the paper, so the language model itself trained reasonably well.

**Diversity (D-1 / D-2).** Ours is 4.0 / 19.6 and the paper's is 9.2 / 39.9. Our replies are short and repetitive. BM25 is more diverse (D-2 27.9) because it reuses real human replies.

**Retrieval (R@10).** Ours finds the right reply in the top 10 36.1% of the time; the paper reports 74.8%. BM25 gets 11.4%. Caveat: we searched among 100 candidates because the paper never states its pool size.

## 3. Ablations (does each component matter?)
- Removing the **constraint** made results slightly *better* (B-2 6.62 vs 6.32, D-2 22.3 vs 19.6).
- Removing the **ResIDs** made almost no difference.
- Only **retrieval-only** (copying without generating) was clearly worse.
- Strategy accuracy is the same across all variants because they share one strategy predictor. That is expected, not a bug.

The paper's Table 6 claims each component contributes. In our re-implementation they don't. Either the paper's implementation differs in ways it doesn't describe, or the gains it reports are small and fragile.

## 4. Open LLMs (Qwen2.5-7B, Mistral-7B)
Ours were far below the paper's numbers: ACC 12–14 against 37–39, and B-2 about 3.9 against 10–11. We ran them zero-shot with the paper's prompt; the paper likely used tuning or few-shot examples it doesn't describe. Our LLM perplexities (96.7, 34.9) were measured differently from the paper's, so don't compare those. SAGE beat the LLMs on ESConv in our setup, but our LLM baseline may be weaker than the paper's.

## 5. ESC-Eval (331 role-play conversations, scored 1–5)
The paper says SAGE beats the LLMs here (overall 4.12 vs 3.80). In our run the LLMs beat SAGE on fluency (3.6 vs 2.4), information (2.9 vs 1.9), skill (2.9–3.3 vs 1.9) and empathy (3.9 vs 3.65). SAGE only edges ahead on humanoid (2.19 vs 2.0). Overall is a tie at about 1.5.

Why: our SAGE is a 220M-parameter T5-base trained only on ESConv; in a 5-turn live conversation it gives short, generic replies, while 7B instruction-tuned models write richer ones.

Setup differences:
- An automatic judge produced every score; the paper's Table 5 used human raters.
- 331 public role cards vs the paper's 2,801 unreleased ones.
- Our judge picks the most likely score in one pass, because the original code crashes on current libraries.
- Overall and Humanoid are low for every system, so they say little.

## 6. Likely causes of the gap
1. No official code; many details are guessed (see AUDIT.md).
2. Unstated settings: retrieval pool size, LLM prompting, perplexity measurement, private ESC-Eval cards.
3. Model size: T5-base is small for open-ended support chat.
4. Judge differences on ESC-Eval.

## 7. One-sentence summary
"I re-implemented SAGE from scratch and ran it on the DGX. It beats retrieval and zero-shot LLM baselines on ESConv, consistent with the paper's direction, but reaches only about half its reported scores. The component ablations and the ESC-Eval advantage over LLMs did not reproduce, which points to undocumented implementation details in the paper."
