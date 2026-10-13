"""Metrics used in the SAGE paper (Sec. 4.2), implemented in one place.

Generation: ACC, Macro-F1 over the 8 strategies, BLEU-2/4, Distinct-1/2,
ROUGE-L (beta=1.2, as in Eq. 10). Retrieval: Recall@K, MRR@10, nDCG@10.
Bias: strategy entropy (log2, since the paper's 2.34 exceeds ln 8) and
Jensen-Shannon divergence from uniform (log2).

The paper does not say whether BLEU/Distinct are corpus- or sentence-level;
we report corpus BLEU (nltk, smoothing method 1) and corpus Distinct, x100.
Perplexity is computed by the model scripts, not here.

Usage: python metrics.py predictions.jsonl
  each line: {"response": gold, "prediction": text, "strategy": gold,
              "pred_strategy": label, optional "ranked_ids": [...], "gold_id": ...}
"""
import json
import math
import sys
from collections import Counter

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from sklearn.metrics import f1_score

from prepare_data import STRATEGIES


def tokenize(text):
    return text.lower().replace(".", " .").replace(",", " ,").replace("?", " ?").replace("!", " !").split()


def distinct(hyps, n):
    grams = [tuple(h[i:i + n]) for h in hyps for i in range(len(h) - n + 1)]
    return 100.0 * len(set(grams)) / max(len(grams), 1)


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hyp, ref, beta=1.2):
    if not hyp or not ref:
        return 0.0
    l = lcs(hyp, ref)
    if l == 0:
        return 0.0
    r, p = l / len(ref), l / len(hyp)
    return (1 + beta ** 2) * r * p / (r + beta ** 2 * p)


def generation_metrics(refs, hyps):
    R = [[tokenize(r)] for r in refs]
    H = [tokenize(h) for h in hyps]
    sm = SmoothingFunction().method1
    return {
        "B-2": 100 * corpus_bleu(R, H, weights=(0.5, 0.5), smoothing_function=sm),
        "B-4": 100 * corpus_bleu(R, H, weights=(0.25,) * 4, smoothing_function=sm),
        "D-1": distinct(H, 1),
        "D-2": distinct(H, 2),
        "R-L": 100 * sum(rouge_l(h, r[0]) for h, r in zip(H, R)) / len(H),
    }


def strategy_metrics(gold, pred):
    acc = 100 * sum(g == p for g, p in zip(gold, pred)) / len(gold)
    per = f1_score(gold, pred, labels=STRATEGIES, average=None, zero_division=0)
    counts = Counter(pred)
    dist = [counts[s] / len(pred) for s in STRATEGIES]
    ent = -sum(p * math.log2(p) for p in dist if p > 0)
    uni = [1 / len(STRATEGIES)] * len(STRATEGIES)
    m = [(p + u) / 2 for p, u in zip(dist, uni)]
    kl = lambda a, b: sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)
    return {
        "ACC": acc,
        "Macro-F1": 100 * per.mean(),
        "Entropy": ent,
        "Bias(JSD)": 0.5 * kl(dist, m) + 0.5 * kl(uni, m),
        "per_strategy_F1": {s: round(100 * f, 2) for s, f in zip(STRATEGIES, per)},
    }


def retrieval_metrics(ranked_lists, golds, ks=(1, 5, 10)):
    out = {f"R@{k}": 0.0 for k in ks}
    mrr = ndcg = 0.0
    for ranked, g in zip(ranked_lists, golds):
        rank = ranked.index(g) + 1 if g in ranked else None
        for k in ks:
            out[f"R@{k}"] += rank is not None and rank <= k
        if rank and rank <= 10:
            mrr += 1 / rank
            ndcg += 1 / math.log2(rank + 1)  # one relevant item, so IDCG = 1
    n = len(golds)
    out = {k: 100 * v / n for k, v in out.items()}
    out["MRR@10"] = mrr / n
    out["nDCG@10"] = ndcg / n
    return out


def evaluate_file(path):
    rows = [json.loads(l) for l in open(path)]
    res = {"n": len(rows)}
    if all("prediction" in r for r in rows):
        res.update(generation_metrics([r["response"] for r in rows], [r["prediction"] for r in rows]))
    if all("pred_strategy" in r for r in rows):
        res.update(strategy_metrics([r["strategy"] for r in rows], [r["pred_strategy"] for r in rows]))
    if all("ranked_ids" in r for r in rows):
        res.update(retrieval_metrics([r["ranked_ids"] for r in rows], [r["gold_id"] for r in rows]))
    return res


if __name__ == "__main__":
    print(json.dumps(evaluate_file(sys.argv[1]), indent=2))
