# ESC-Eval role cards

From the official ESC-Eval repository, https://github.com/haidequanbu/ESC-Eval (commit 9ad46e7), `data/`:
Zhao et al., "ESC-Eval: Evaluating Emotion Support Conversations in Large Language Models", EMNLP 2024.

- `card_high_en.json`: 331 English high-quality role cards (sources: ESConv 158, MHP 73, ExTES 70, Psych 25, EPITOME 5)
- `card_high_zh.json`: 324 Chinese high-quality role cards (PsyQA 252, smile 72)
- `escrank_prompts_en.json`: the seven English scoring prompts from ESC-Eval `score.py` (used with ESC-RANK)

The repository has no license file; included here for academic research use only, with attribution.
These 655 cards are the "high-quality" set ESC-Eval evaluates on. The 2,801-card pool the SAGE paper cites (Table A.10) is not public.
