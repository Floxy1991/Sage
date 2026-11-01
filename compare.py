"""Our results next to the numbers SAGE reports (Tables 1, 2, 4 and 5), as readable tables.

Usage: python compare.py            (reads results/*/metrics.json, retrieval.json and esceval.json)
In a notebook:  import compare; compare.show()     (renders the tables; needs pandas)
Each table puts the paper's row first, then our runs of the same system. Runs with no results are left out.
"""
import glob
import json
import math
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
GEN_COLS = ["ACC", "Macro-F1", "PPL", "B-2", "B-4", "D-1", "D-2", "R-L"]
RET_COLS = ["R@1", "R@10", "MRR@10", "nDCG@10"]
ESC_PAPER = {  # SAGE Table 5 (human raters, 5-point scale); Exp./Skill./Overall from Table C.15 "Overall" row
    "SAGE": {"Emp.": 4.12, "Flu.": 4.39, "Hum.": 4.28, "Info.": 4.05, "Exp.": 4.28, "Skill.": 4.08, "Overall": 4.12},
    "Qwen-2.5-7B": {"Emp.": 3.84, "Flu.": 4.28, "Hum.": 4.12, "Info.": 3.76, "Exp.": 3.91, "Skill.": 3.74, "Overall": 3.80},
    "Mistral-7B": {"Emp.": 3.62, "Flu.": 4.18, "Hum.": 3.95, "Info.": 3.71},
    "LLaMA-3.1-8B": {"Emp.": 3.54, "Flu.": 4.12, "Hum.": 3.87, "Info.": 3.65},
}
ESC_COLS = ["Flu.", "Exp.", "Emp.", "Info.", "Hum.", "Skill.", "Overall"]
GUESS = {"qwen": "Qwen-2.5-7B", "llama": "LLaMA-3.1-8B", "mistral": "Mistral-7B", "sage": "SAGE", "blender": "BlenderBot-Joint"}


def _num(v):
    return v if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None


def collect(results="results"):
    """[(run name, metrics dict, paper system or None)] for every results/<run>/ folder."""
    runs = []
    for d in sorted(glob.glob(os.path.join(results, "*/"))):
        run = os.path.basename(d.rstrip("/\\"))
        m = {}
        for name in ("metrics.json", "retrieval.json"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                j = json.load(open(p, encoding="utf-8"))
                m.update(j.get("generation", {}), **j.get("strategy", {}), **j.get("retrieval", {}))
                m.update({k: v for k, v in j.items() if isinstance(v, (int, float))})
        esc = os.path.join(d, "esceval.json")
        if os.path.exists(esc):
            j = json.load(open(esc, encoding="utf-8"))
            m["esc"] = dict(j.get("sage_scale_1to5", {}), n=j.get("n"), judge=j.get("judge", ""))
        key = run.lower().replace("esceval_", "")
        paper = next((v for k, v in GUESS.items() if k in key), None)
        runs.append((run, m, paper))
    return runs


def rows(runs, cols, paper_table, get=lambda m: m):
    """Table rows: per paper system, its paper row then our matching runs; then runs with no paper match."""
    out, order = [], list(paper_table) + [None]
    for system in order:
        mine = [(r, get(m)) for r, m, p in runs if p == system and get(m)
                and any(_num(get(m).get(c)) is not None for c in cols)]
        if not mine:
            continue
        if system is not None:
            ref = paper_table[system]
            out.append({"system": system, "source": "paper", "n": None, **{c: ref.get(c) for c in cols}})
        for r, m in mine:
            out.append({"system": r, "source": "ours", "n": m.get("n"), **{c: _num(m.get(c)) for c in cols}})
    return out


def build(results="results"):
    """{title: list of row dicts} for the generation, retrieval and ESC-Eval tables."""
    runs = collect(results)
    paper_ret = {k: v for k, v in PAPER.items() if any(c in v for c in RET_COLS)}
    return {
        "Generation and strategy (Tables 1-2)": rows(runs, GEN_COLS, PAPER),
        "Retrieval (Table 4)": rows(runs, RET_COLS, paper_ret),
        "ESC-Eval, 1-5 scale (Table 5; ours: automatic judge, paper: human raters)":
            rows(runs, ESC_COLS, ESC_PAPER, get=lambda m: m.get("esc", {})),
    }


def fmt(v):
    v = _num(v)
    return "" if v is None else str(v) if isinstance(v, int) else f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"


def markdown(tables=None):
    lines = []
    for title, rs in (tables or build()).items():
        if not rs:
            continue
        cols = [c for c in rs[0] if c not in ("system", "source")]
        lines += [f"### {title}", "", "| system | source | " + " | ".join(cols) + " |",
                  "|---|---|" + "---:|" * len(cols)]
        lines += ["| " + " | ".join([r["system"], r["source"]] + [fmt(r[c]) for c in cols]) + " |" for r in rs]
        lines.append("")
    return "\n".join(lines) if lines else "No results yet: run some experiments first."


def show(results="results", save="results/comparison.md"):
    """Render the tables in Jupyter/Colab (pandas), and save them as Markdown."""
    import pandas as pd
    from IPython.display import Markdown, display
    tables = build(results)
    for title, rs in tables.items():
        if not rs:
            continue
        df = pd.DataFrame(rs).set_index(["system", "source"])
        df["n"] = df["n"].astype("Int64")
        display(Markdown(f"**{title}**"))
        fmts = {c: "{:.3f}" if c in ("MRR@10", "nDCG@10", "B-4") else "{:.2f}" for c in df.columns if c != "n"}
        display(df.style.format(fmts, na_rep="").set_properties(**{"text-align": "right"}))
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        open(save, "w", encoding="utf-8").write(markdown(tables))
        print("Saved", save)


if __name__ == "__main__":
    print(markdown())
