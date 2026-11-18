Subject: SAGE reproduction – progress update

Dear [Supervisor's name],

Here is where I am with reproducing the SAGE paper (Yang et al., ESWA 2026).

**Done so far**
- The authors did not release code ("available on request"), so I built a full re-implementation of SAGE from the paper's description: data preparation, metrics, the self-retrieval model, the ResID index, the fusion generator, inference, retrieval evaluation and the Table 6 ablations. It also covers the second dataset, ESC-Eval.
- Thank you for the DGX access. I trained SAGE there with the paper's full settings, including the **100 self-retrieval epochs**, and evaluated it on the full ESConv test set (3,481 examples, the paper's 7:1:2 split).
- I audited the paper against the public data and found several inconsistencies. For example, the strategy distribution in Table A.9 does not match ESConv, and the "Macro-F1" in Table 1 is actually accuracy. Details are in AUDIT.md.

**Results (full test set, 100 epochs)**

| | Strategy ACC | BLEU-2 | ROUGE-L | PPL | R@10 |
|---|---|---|---|---|---|
| SAGE, paper | 46.8 | 11.87 | 20.51 | 14.32 | 74.8 |
| SAGE, ours | 22.9 | 6.32 | 15.54 | 16.60 | 36.1 |
| BM25 baseline, ours | 21.6 | 4.45 | 12.76 | – | 11.4 |

- Our SAGE clearly beats the BM25 baseline, and it is roughly level with the paper's BlenderBot baseline (BLEU-2 6.18, PPL 18.11). However, it reaches only about half of the scores the paper reports for SAGE.
- The ablations (Table 6) do not reproduce the paper's pattern. Removing the strategy constraint or the ResIDs does not lower the scores (without the constraint, BLEU-2 is slightly higher at 6.62). Only the retrieval-only variant is clearly worse (BLEU-2 4.27).
- The paper leaves out several settings that matter: the size of the retrieval candidate pool, several hyperparameters, and the prompts for the LLM baselines. I documented each assumption I made. Our SAGE retrieval scores use a pool of 100 candidates, so they depend on that choice.

My reading is that the gap is too large to be explained by these missing details alone. Unless the authors share their code or settings, the paper's headline numbers cannot be reproduced from its description.

**Next steps**
1. Run the open LLM baselines (Qwen2.5-7B, Mistral-7B) on the DGX with the same test set (in progress).
2. Run the ESC-Eval interactive evaluation.
3. Write up all the tables and the audit in the final report.

It might also be worth contacting the authors to ask for their code or exact settings, since the paper says these are "available on request". I would welcome your view on that.

Best regards,
Charles
