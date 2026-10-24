#!/usr/bin/env python3
"""Render the held-out accuracy chart from results/ (strict = right label, bare reply)."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse import load, acc

MODELS = [("ps-gemma-3-4b-it", "Gemma 3 4B"), ("ps-gemma-3-27b-it", "Gemma 3 27B")]
PROMPTS = [("lazy", "v1: labels only", "#9aa5b1"), ("fixed", "v2: nap 1", "#5b8def"), ("fixed2", "v3: nap 2", "#1f5fbf")]

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
w = 0.26
for i, (alias, label, color) in enumerate(PROMPTS):
    for j, (model, _) in enumerate(MODELS):
        vals = [acc(r, strict=True) * 100 for r in load(alias, model)]
        mean = sum(vals) / len(vals)
        x = j + (i - 1) * w
        ax.bar(x, mean, w * 0.92, color=color, label=label if j == 0 else None)
        ax.vlines(x, min(vals), max(vals), color="#1b2733", lw=1.4)
        ax.text(x, mean + 2.2, f"{mean:.0f}%", ha="center", fontsize=11, color="#1b2733")
ax.set_xticks(range(len(MODELS)), [m[1] for m in MODELS], fontsize=12)
ax.set_ylim(0, 85)
ax.set_ylabel("Correct, parseable replies (%)", fontsize=11)
ax.set_title("Banking77 held-out test: 308 messages the optimizer never saw\n(bar = mean of 2 runs, line = range)",
             fontsize=12, loc="left", color="#1b2733")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e5e8eb")
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=10)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "heldout-accuracy.png")
fig.savefig(out)
print(out)
