from decimal import Decimal, ROUND_HALF_UP


from pathlib import Path


import re


import numpy as np


TASKS = {
    "binary": "Bin",
    "classification": "MultiC",
    "regression": "Reg",
    "multiregression": "MultiReg",
}


def tex_escape(text):
    special = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "&": "\\&",
        "%": "\\%",
        "#": "\\#",
        "$": "\\$",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join((special.get(c, c) for c in str(text)))


def percentage(value, digits=2, zero_dash=True):
    if not np.isfinite(value):
        raise ValueError("Missing percentages cannot be rendered as zero")
    if zero_dash and value == 0:
        return "-"
    return str(
        Decimal(str(value)).quantize(Decimal(10) ** (-digits), rounding=ROUND_HALF_UP)
    )


def render(template, **values):
    text = Path(template).read_text()
    for key, value in values.items():
        text = text.replace(f"<<{key.upper()}>>", str(value))
    if re.search("<<[A-Z_]+>>", text):
        raise ValueError("Unfilled table template")
    return text


def domain_rows(frame, domains):
    columns = list(TASKS)
    rows = []
    n = len(frame)
    if not n:
        raise ValueError("Dataset scope is empty")
    for domain in domains:
        subset = frame[frame.domain.eq(domain["summary_domain"])]
        counts = subset.task.value_counts()
        cells = [domain["display_tex"], str(len(subset))]
        cells += [
            str(int(counts.get(task))) if counts.get(task, 0) else "-"
            for task in columns
        ]
        cells += [percentage(100 * len(subset) / n, digits=1, zero_dash=False) + "\\%"]
        rows.append(" & ".join(cells) + " \\\\")
    totals = frame.task.value_counts()
    rows += [
        "\\midrule",
        " & ".join(
            ["Total", str(n)]
            + [str(int(totals.get(t, 0))) for t in columns]
            + ["100\\%"]
        )
        + " \\\\",
    ]
    return rows


def overlap_rows(overlaps, models=("iLTM", "HyperFast", "TabDPT", "RealTabPFNv2.5")):
    groups = {k: set(v) for k, v in overlaps.items() if k != "all"}
    if set(models) != set(groups):
        raise ValueError("Overlap model columns differ from the verified source")
    union = set().union(*groups.values())
    if union != set(overlaps["all"]):
        raise ValueError("Overlap union differs from per-model membership")
    rows = []
    for dataset in sorted(union):
        cells = ["\\texttt{" + tex_escape(dataset) + "}"]
        cells += ["\\cmark" if dataset in groups[m] else "" for m in models]
        rows.append(" & ".join(cells) + " \\\\")
    rows += [
        "\\midrule",
        " & ".join(["Total"] + [str(len(groups[m])) for m in models]) + " \\\\",
    ]
    return rows
