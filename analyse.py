#!/usr/bin/env python3
"""Summarise the held-out runs: lenient (routing) and strict (bare-label) accuracy per
prompt and model, averaged over repeats, plus a paired exact McNemar test per change."""
import glob, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["ps-gemma-3-4b-it", "ps-gemma-3-27b-it"]
ALIASES = ["lazy", "fixed", "fixed2"]


def load(alias, model):
    runs = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", f"eval-test-{alias}-{model}-r[12].json"))):
        rows = json.load(open(f))["rows"]
        runs.append({r["text"]: r for r in rows if "error" not in r})
    return runs


def acc(run, strict):
    ok = [r["correct"] and (r["well_formed"] or not strict) for r in run.values()]
    return sum(ok) / len(ok)


def mcnemar(a, b, strict):
    """Exact two-sided McNemar on messages scored in both runs (repeat 1 of each)."""
    win = lose = 0
    for t in a.keys() & b.keys():
        x = a[t]["correct"] and (a[t]["well_formed"] or not strict)
        y = b[t]["correct"] and (b[t]["well_formed"] or not strict)
        win += y and not x
        lose += x and not y
    n, k = win + lose, min(win, lose)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return win, lose, p


if __name__ == "__main__":
    data = {(a, m): load(a, m) for a in ALIASES for m in MODELS}
    for m in MODELS:
        print(f"\n{m}")
        for a in ALIASES:
            runs = data[(a, m)]
            for strict in (False, True):
                vals = [acc(r, strict) for r in runs]
                tag = "strict " if strict else "lenient"
                cost = sum(r2["cost_usd"] for r2 in runs[0].values()) / len(runs[0]) * 1000
                print(f"  {a:6} {tag} " + " / ".join(f"{v:.3f}" for v in vals)
                      + f"  mean {sum(vals)/len(vals):.3f}" + (f"  ${cost:.3f}/1k" if strict else ""))
        for base, new in (("lazy", "fixed"), ("fixed", "fixed2"), ("lazy", "fixed2")):
            for strict in (False, True):
                w, l, p = mcnemar(data[(base, m)][0], data[(new, m)][0], strict)
                print(f"  {base}->{new} {'strict ' if strict else 'lenient'}: +{w} -{l}  p={p:.4f}")
