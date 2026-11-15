#!/usr/bin/env python3
"""Render tables/reported_values_v2.yaml to CSV, Markdown and a LaTeX fragment.

usage: generate_tables_v2.py [--out tables/generated_v2]
Every output row carries evidence class and software-reproducibility flag so a
reader can see which numbers need the original hardware.
"""
import argparse, csv
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def tex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("^", r"\^{}"), ("<=", r"$\leq$"), (">=", r"$\geq$"),
                 ("+-", r"$\pm$")):
        s = s.replace(a, b)
    return s


def write_csv(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return cols


def md_table(cols, rows):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def tex_table(cols, rows, caption, label):
    spec = "l" * len(cols)
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             rf"\caption{{{tex_escape(caption)}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & ".join(tex_escape(c.replace("_", " ")) for c in cols) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(tex_escape(r.get(c, "")) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "tables" / "reported_values_v2.yaml"))
    ap.add_argument("--out", default=str(ROOT / "tables" / "generated_v2"))
    a = ap.parse_args()
    data = yaml.safe_load(open(a.src))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, tex = ["# Chapter 4 reported values (generated)\n"], []
    for key, blk in data.items():
        if isinstance(blk, dict) and "rows" in blk:
            title, rows = blk["title"], blk["rows"]
        elif isinstance(blk, list):
            title, rows = key, blk
        else:
            continue
        rows = [{k: ("yes" if v is True else "no" if v is False else v) for k, v in r.items()} for r in rows]
        cols = write_csv(out / f"{key}_v2.csv", rows)
        md += [f"## {key}: {title}\n", md_table(cols, rows), ""]
        tex.append(tex_table(cols, rows, f"{key}: {title}", f"tab:v2:{key}"))
    (out / "chapter4_tables_v2.md").write_text("\n".join(md))
    (out / "chapter4_tables_v2_fragment.tex").write_text("\n".join(tex))
    print("wrote", len(list(out.glob('*'))), "files to", out)


if __name__ == "__main__":
    main()
