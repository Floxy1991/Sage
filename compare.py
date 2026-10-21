"""Print our results next to the numbers SAGE reports (Tables 1, 2 and 4).

Usage: python compare.py            (reads results/*/metrics.json and results/*/retrieval.json)
"""
import glob
import json
import os

# SAGE paper, Table 1 (ACC, PPL, B-2, B-4, D-1, D-2, R-L) and Table 2 (Macro-F1).
PAPER = {
    "BlenderBot-Joint": dict(ACC=27.72, PPL=18.11, **{"B-2": 6.18, "B-4": 1.66, "D-1": 3.27, "D-2": 20.87, "R-L": 15.13}),
    "Qwen-2.5-7B": dict(ACC=36.89, PPL=14.58, **{"B-2": 10.89, "B-4": 3.24, "D-1": 8.12, "D-2": 37.86, "R-L": 19.21, "Macro-F1": 29.99}),
    "LLaMA-3.1-8B": dict(ACC=37.45, PPL=15.48, **{"B-2": 9.32, "B-4": 2.76, "D-1": 6.42, "D-2": 33.67, "R-L": 18.34, "Macro-F1": 32.89}),
    "Mistral-7B": dict(ACC=38.91, PPL=15.12, **{"B-2": 9.87, "B-4": 2.94, "D-1": 6.85, "D-2": 35.14, "R-L": 18.92, "Macro-F1": 31.39}),
    "SAGE": dict(ACC=46.80, PPL=14.32, **{"B-2": 11.87, "B-4": 3.88, "D-1": 9.24, "D-2": 39.86, "R-L": 20.51, "Macro-F1": 38.95,
                                            "R@1": 31.6, "R@10": 74.8, "MRR@10": 0.437, "nDCG@10": 0.703}),
}
COLS = ["ACC", "Macro-F1", "PPL", "B-2", "B-4", "D-1", "D-2", "R-L", "R@1", "R@10", "MRR@10", "nDCG@10"]
ESC_PAPER = {  # SAGE Table 5 (human raters, 5-point scale); Exp./Skill./Overall from Table C.15 "Overall" row
    "SAGE": {"Emp.": 4.12, "Flu.": 4.39, "Hum.": 4.28, "Info.": 4.05, "Exp.": 4.28, "Skill.": 4.08, "Overall": 4.12},
    "Qwen-2.5-7B": {"Emp.": 3.84, "Flu.": 4.28, "Hum.": 4.12, "Info.": 3.76, "Exp.": 3.91, "Skill.": 3.74, "Overall": 3.80},
    "Mistral-7B": {"Emp.": 3.62, "Flu.": 4.18, "Hum.": 3.95, "Info.": 3.71},
    "LLaMA-3.1-8B": {"Emp.": 3.54, "Flu.": 4.12, "Hum.": 3.87, "Info.": 3.65},
}
ESC_COLS = ["Flu.", "Exp.", "Emp.", "Info.", "Hum.", "Skill.", "Overall"]
GUESS = {"qwen": "Qwen-2.5-7B", "llama": "LLaMA-3.1-8B", "mistral": "Mistral-7B", "sage": "SAGE", "blender": "BlenderBot-Joint"}


def fmt(v):
    return "" if v is None else f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"


def main():
    rows = []
    for d in sorted(glob.glob("results/*/")):
        m = {}
        for name in ("metrics.json", "retrieval.json"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                j = json.load(open(p))
                m.update(j.get("generation", {}), **j.get("strategy", {}), **j.get("retrieval", {}))
                m.update({k: v for k, v in j.items() if isinstance(v, (int, float))})
        run = os.path.basename(d.rstrip("/"))
        paper = None if run.lower().startswith("sage_") else next((v for k, v in GUESS.items() if k in run.lower()), None)
        rows.append((run, m, paper))
    print("| run | n | " + " | ".join(COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for run, m, paper in rows:
        print(f"| {run} (ours) | {m.get('n', '')} | " + " | ".join(fmt(m.get(c)) for c in COLS) + " |")
        if paper:
            print(f"| {paper} (paper) | | " + " | ".join(fmt(PAPER[paper].get(c)) for c in COLS) + " |")

    esc = sorted(glob.glob("results/esceval_*/esceval.json"))
    if esc:
        print("\nESC-Eval (ours: ESC-RANK/LLM judge, 0-4 shifted to 1-5; paper: human raters, 5-point)\n")
        print("| run | n | judge | " + " | ".join(ESC_COLS) + " |")
        print("|---|---|---|" + "---|" * len(ESC_COLS))
        for p in esc:
            j = json.load(open(p))
            run = os.path.basename(os.path.dirname(p))
            print(f"| {run} (ours) | {j['n']} | {j.get('judge', '')} | "
                  + " | ".join(fmt(j["sage_scale_1to5"].get(c)) for c in ESC_COLS) + " |")
            paper = next((v for k, v in GUESS.items() if k in run.lower().replace("esceval_", "")), None)
            if paper in ESC_PAPER:
                print(f"| {paper} (paper) | | human | " + " | ".join(fmt(ESC_PAPER[paper].get(c)) for c in ESC_COLS) + " |")


if __name__ == "__main__":
    main()
