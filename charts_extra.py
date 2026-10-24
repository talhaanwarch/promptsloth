#!/usr/bin/env python3
"""Two more charts from results/: what each 4B answer was, and judge vs held-out."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyse import load

OUT = os.path.join(HERE, "images")
INK, GRID = "#1b2733", "#e5e8eb"
PROMPTS = [("lazy", "v1: labels only"), ("fixed", "v2: after iteration 1"), ("fixed2", "v3: after iteration 2")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


# 1. Where each Gemma 3 4B answer landed (mean of 2 runs, % of 308)
parts = {"Right label, clean reply": [], "Right label, extra characters": [], "Wrong label": []}
for alias, _ in PROMPTS:
    runs = load(alias, "ps-gemma-3-4b-it")
    clean = sum(sum(r["correct"] and r["well_formed"] for r in run.values()) / len(run) for run in runs) / len(runs)
    messy = sum(sum(r["correct"] and not r["well_formed"] for r in run.values()) / len(run) for run in runs) / len(runs)
    parts["Right label, clean reply"].append(clean * 100)
    parts["Right label, extra characters"].append(messy * 100)
    parts["Wrong label"].append((1 - clean - messy) * 100)
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
colors = ["#1f5fbf", "#f2a541", "#d5dae0"]
left = [0, 0, 0]
labels = [p[1] for p in PROMPTS]
for (name, vals), c in zip(parts.items(), colors):
    ax.barh(labels, vals, left=left, color=c, label=name, height=0.6)
    for i, v in enumerate(vals):
        if v >= 4:
            ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=11,
                    color="white" if c == "#1f5fbf" else INK)
    left = [l + v for l, v in zip(left, vals)]
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Share of 308 held-out messages (mean of 2 runs)")
ax.set_title("Gemma 3 4B: iteration 1 fixed the format, iteration 2 turned wrong labels into right ones",
             loc="left", fontsize=12, color=INK)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=10)
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "answer-breakdown-v2.png"))

# 2. Judge on training messages vs exact match on held-out messages
judge = []
for f, key in (("optimize-run1-report.json", "baseline"), ("optimize-run2-report.json", "baseline"),
               ("optimize-run2-report.json", "winner")):
    r = json.load(open(os.path.join(HERE, "results", f)))
    if key == "baseline":
        judge.append(r["baseline"][0]["avgScore"])
    else:
        judge.append(next(c for c in r["candidates"] if c["candidateId"] == r["recommended"])["scores"][0]["avgScore"])
held = []
for alias, _ in PROMPTS:
    runs = load(alias, "ps-gemma-3-4b-it")
    held.append(100 * sum(sum(r["correct"] and r["well_formed"] for r in run.values()) / len(run) for run in runs) / len(runs))
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
x = range(3)
ax.plot(x, judge, marker="o", lw=2.5, color="#7a5af5", label="Open judge (Gemma 4 31B), 154 training messages")
ax.plot(x, held, marker="s", lw=2.5, color="#1f5fbf", label="Exact match, 308 held-out messages, no judge")
for i in x:
    ax.text(i, judge[i] + 2.5, f"{judge[i]:.0f}", ha="center", color="#7a5af5", fontsize=11)
    ax.text(i, held[i] - 5.5, f"{held[i]:.0f}%", ha="center", color="#1f5fbf", fontsize=11)
ax.set_xticks(list(x), [p[1] for p in PROMPTS], fontsize=11)
ax.set_ylim(20, 80)
ax.set_ylabel("Score / accuracy, Gemma 3 4B")
ax.set_title("The judge's training scores ranked the prompts the same way as the held-out test",
             loc="left", fontsize=12, color=INK)
ax.grid(axis="y", color=GRID)
ax.legend(frameon=False, loc="lower right", fontsize=10)
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "judge-vs-heldout-v2.png"))
print([round(v, 1) for v in judge], [round(v, 1) for v in held], {k: [round(v, 1) for v in vs] for k, vs in parts.items()})
