"""CPU-only sanity baseline: BM25 retrieval over ESConv.

Two uses:
  1. Retrieval eval (Tables 4/8 style). Query = last 3 context turns; the
     candidate pool = every gold supporter response in the test split; the
     correct item = this turn's own gold response. (SAGE never defines its
     pool or "correct response", so this is our stated assumption.)
  2. Generation eval. Return the top BM25 training response and its strategy,
     as a no-learning floor for Table 1.

Usage: python bm25_baseline.py --split paper --out results/bm25_paper
"""
import argparse
import json
import os

from rank_bm25 import BM25Okapi

from metrics import generation_metrics, retrieval_metrics, strategy_metrics, tokenize


def load(path):
    return [json.loads(l) for l in open(path)]


def query_text(ex, turns=3):
    return " ".join(t["text"] for t in ex["context"][-turns:])


def bucket(n_prev):
    # Table 8 buckets count dialogue turns; one turn = a seeker+supporter pair.
    turns = (n_prev + 1) // 2
    return "1-2" if turns <= 2 else "3-4" if turns <= 4 else "5-6" if turns <= 6 else ">=7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="paper")
    ap.add_argument("--out", default="results/bm25")
    args = ap.parse_args()
    train = load(f"data/{args.split}/train.jsonl")
    test = load(f"data/{args.split}/test.jsonl")
    os.makedirs(args.out, exist_ok=True)

    # 1. Retrieval over the test response pool.
    pool = BM25Okapi([tokenize(e["response"]) for e in test])
    ranked = []
    for e in test:
        scores = pool.get_scores(tokenize(query_text(e)))
        ranked.append(sorted(range(len(test)), key=lambda i: -scores[i])[:10])
    golds = list(range(len(test)))
    ret = retrieval_metrics(ranked, golds)
    by_len = {}
    for b in ("1-2", "3-4", "5-6", ">=7"):
        idx = [i for i, e in enumerate(test) if bucket(e["num_prev_turns"]) == b]
        by_len[b] = {"n": len(idx), "R@10": retrieval_metrics([ranked[i] for i in idx], idx)["R@10"]}

    # 2. Nearest training response as the generated reply.
    index = BM25Okapi([tokenize(query_text(e)) for e in train])
    preds = []
    for e in test:
        scores = index.get_scores(tokenize(query_text(e)))
        best = train[max(range(len(train)), key=lambda i: scores[i])]
        preds.append({"id": e["id"], "response": e["response"], "strategy": e["strategy"],
                      "prediction": best["response"], "pred_strategy": best["strategy"]})
    with open(os.path.join(args.out, "predictions.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    gen = generation_metrics([p["response"] for p in preds], [p["prediction"] for p in preds])
    strat = strategy_metrics([p["strategy"] for p in preds], [p["pred_strategy"] for p in preds])
    res = {"split": args.split, "n_test": len(test), "retrieval": ret, "retrieval_R@10_by_turns": by_len,
           "generation": gen, "strategy": strat}
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
