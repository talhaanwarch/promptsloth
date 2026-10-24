#!/usr/bin/env python3
"""
PromptSloth: a lazy prompt fixes itself on open-weight models, over MCP only. 🦥

Every step talks to AcruxCore through its MCP server (JSON-RPC at /api/v1/mcp):

  setup     create_prompt + commit_prompt_version  -> the lazy v1 (labels only)
  dataset   create_dataset + add_dataset_example   -> 154 Banking77 *train* messages
  optimize  optimize_prompt + get_optimize_report  -> the optimizer drafts and judges
  promote   promote_optimize_candidate             -> the winner gets its own alias
  eval      run_prompt per message                 -> exact-match accuracy on the
                                                      308 held-out *test* messages

The held-out score uses no judge: the model's reply either equals the label or it
does not. The optimizer never sees a test message.

    export ACRUXCORE_API_KEY=...            # editor role; never commit it
    python3 sample.py                       # writes data/splits.json
    python3 promptsloth.py setup                                   # v1, alias `lazy`
    # iteration 1: format
    python3 promptsloth.py dataset --round 1                       # prints the dataset id
    python3 promptsloth.py optimize --dataset <id> --baseline lazy \
        --model ps-gemma-3-4b-it --optimizer ps-gpt-oss-120b --judge rs-gemma-4-31b
    python3 promptsloth.py promote --run <run_id> --candidate <candidate_id> --alias fixed
    # iteration 2: routing definitions
    python3 promptsloth.py dataset --round 2
    python3 promptsloth.py optimize --dataset <id> --baseline fixed \
        --model ps-gemma-3-4b-it --optimizer ps-gpt-oss-120b --judge rs-gemma-4-31b
    python3 promptsloth.py promote --run <run_id> --candidate <candidate_id> --alias fixed2
    # held-out test, per alias, model and repeat; then the tables and chart
    python3 promptsloth.py eval --alias fixed2 --model ps-gemma-3-4b-it --repeat 1
    python3 analyse.py && python3 chart.py
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import mcp

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = "promptsloth-banking77"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def splits():
    with open(os.path.join(HERE, "data", "splits.json"), encoding="utf-8") as f:
        return json.load(f)


def lazy_v1(intents):
    """The prompt a busy developer writes first: the label list and nothing else."""
    return [{
        "role": "system",
        "content": ("Classify the bank customer's message into exactly one of these intents: "
                    + ", ".join(intents)
                    + ". Reply with the intent label only, nothing else."),
    }, {"role": "user", "content": "{{ message }}"}]


OVERALL_FEEDBACK = (
    "Each answer must be exactly one intent label from the list, spelled exactly as listed, "
    "with nothing else. Many labels sound alike, so the prompt must say what separates them: "
    "for each group of similar intents, state the deciding detail (for example, a card that "
    "has not arrived yet versus one that must be activated; a payment that is pending versus "
    "one that was declined; a top-up versus a transfer; a fee versus an exchange rate "
    "question). Write those rules from the examples. Keep a system message with the label "
    "list and keep the {{ message }} placeholder in the user message."
)


# Round 2: the round-1 rewrite fixed the output format but added only generic rules, so
# held-out routing barely moved. This rubric asks for the missing policy itself.
ROUND2_FEEDBACK = (
    "The output format is already correct; keep it. The remaining errors are wrong intents "
    "between labels that sound alike. Rewrite the prompt so that EVERY one of the 77 labels "
    "has a one-line definition saying what the customer is asking about and how it differs "
    "from its nearest neighbour (for example: card_arrival = ordered card has not arrived; "
    "card_delivery_estimate = asks how long delivery takes before it is late). Derive the "
    "definitions from the example messages. Keep the label spelling exact, keep the reply "
    "to the bare label, and keep the {{ message }} placeholder in the user message."
)


# ------------------------------------------------------------------ steps
def setup(_args):
    if not mcp.tool("get_prompt", {"prompt": PROMPT})[1]:
        print(f"{PROMPT} already exists; not committing a second v1.")
        return
    print(mcp.tool("create_prompt", {"name": PROMPT,
                   "description": "PromptSloth: Banking77 intent router. v1 knows only the labels."})[0][:200])
    print(mcp.tool("commit_prompt_version", {"prompt": PROMPT, "messages": lazy_v1(splits()["intents"])})[0][:300])
    print(mcp.tool("promote_prompt_alias", {"prompt": PROMPT, "alias": "lazy", "version_number": 1})[0][:200])


def dataset(args):
    data = splits()
    name = f"promptsloth-banking77-train-r{args.round}-{int(time.time())}"
    feedback = OVERALL_FEEDBACK if args.round == 1 else ROUND2_FEEDBACK
    ds = mcp.tool_json("create_dataset", {"name": name, "overall_feedback": feedback})
    ds_id = ds.get("id") or UUID_RE.search(json.dumps(ds)).group(0)

    def add(row):
        text, err = mcp.tool("add_dataset_example", {
            "dataset_id": ds_id, "input": {"message": row["text"]},
            "criteria": (f"The correct intent is {row['intent']}. Full marks only when the reply "
                         f"is exactly the label {row['intent']} and nothing else; zero otherwise."),
        })
        if err:
            raise RuntimeError(text)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(add, data["train"]))
    print(f"dataset {name} ({ds_id}): {len(data['train'])} examples")


def optimize(args):
    started, err = mcp.tool("optimize_prompt", {
        "prompt": PROMPT, "alias": args.baseline, "dataset_id": args.dataset, "models": [args.model],
        "optimizer_model": args.optimizer, "judge_model": args.judge, "draft_count": args.drafts,
    })
    if err:
        raise SystemExit(started)
    run_id = UUID_RE.search(started).group(0)
    print(f"optimize run {run_id} queued. The optimizer is working; go have a rest 🦥")
    last = None
    while True:
        text, _ = mcp.tool("get_optimize_report", {"run_id": run_id})
        status = next((s for s in ("succeeded", "failed", "running", "queued") if s in text[:300]), None)
        if status != last:
            print(" ", time.strftime("%H:%M:%S"), status)
            last = status
        if status in ("succeeded", "failed"):
            break
        time.sleep(30)
    out = os.path.join(HERE, "results", f"optimize-{run_id}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(text)
    print(text[:4000])


def report(args):
    """Print one iteration's scores, straight from get_optimize_report."""
    r = mcp.tool_json("get_optimize_report", {"run_id": args.run})
    print(f"optimize run {args.run[:8]}  judge: {r['judgeModel']}  status: {r['status']}")
    for b in r["baseline"]:
        print(f"  {b['label'] + ' (baseline)':22} {b['avgScore']:5.1f}")
    for c in r["candidates"]:
        s = c["scores"][0]
        mark = "  <- recommended" if c["candidateId"] == r.get("recommended") else ""
        print(f"  {c['label']:22} {s['avgScore']:5.1f}  beats baseline: {c['beatsBaseline']}{mark}")


def promote(args):
    text, err = mcp.tool("promote_optimize_candidate",
                         {"run_id": args.run, "candidate_id": args.candidate, "alias": args.alias})
    print(text[:1500])
    if err:
        raise SystemExit(1)


def normalise(reply):
    """First line of the reply, lower-cased, with wrapping quotes, brackets and markdown
    removed. Lenient on purpose: the score measures routing, not formatting."""
    line = (reply or "").strip().split("\n")[0]
    return line.strip().strip("`'\"*.{}[]()<> ").lower()


def well_formed(reply):
    """True when the raw reply is the bare label, so a strict parser would accept it."""
    return bool(re.fullmatch(r"[a-z_?]+", (reply or "").strip().lower()))


def eval_split(args):
    data = splits()
    rows = data[args.split][: args.limit or None]
    valid = {i.lower() for i in data["intents"]}

    def run(row):
        empty = 0
        for attempt in range(8):
            text, err = mcp.tool("run_prompt", {
                "prompt": PROMPT, "alias": args.alias, "model": args.model,
                "variables": {"message": row["text"]}, "temperature": 0, "max_tokens": 40,
                "trace_name": f"promptsloth-eval-{args.alias}",
            })
            if not err:
                # The tool's text is a summary line, then the completion and gateway headers as JSON.
                body = json.JSONDecoder().raw_decode(text[text.index("\n\n{") + 2:])[0]
                content = body["completion"]["choices"][0]["message"]["content"]
                if content and content.strip():
                    break
                # One upstream provider answers some calls with null content; a retry is routed
                # to another. Counted, never scored as a wrong route.
                empty += 1
            time.sleep(2 * (attempt + 1))
        else:
            return {**row, "error": text[:300], "empty_retries": empty}
        gw = body.get("gateway", {})
        answer = normalise(content)
        return {**row, "raw": content, "answer": answer, "correct": answer == row["intent"].lower(),
                "valid_label": answer in valid, "well_formed": well_formed(content),
                "trace_id": gw.get("trace-id"), "provider_model": body["completion"].get("model"),
                "cost_usd": float(gw.get("cost-usd") or 0), "cache": gw.get("cache"),
                "tokens": body["completion"].get("usage", {}).get("total_tokens"), "empty_retries": empty}

    def run_and_show(row):
        r = run(row)
        if args.show:
            mark = "error" if "error" in r else ("✓" if r["correct"] else "✗")
            raw = (r.get("raw") or "").strip().split("\n")[0]
            print(f"  {mark} {row['text'][:50]:50} -> {raw[:40]}", flush=True)
        return r

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_and_show, rows))
    scored = [r for r in results if "error" not in r]
    correct = sum(r["correct"] for r in scored)
    summary = {"alias": args.alias, "model": args.model, "split": args.split, "repeat": args.repeat,
               "n": len(rows), "errors": len(rows) - len(scored), "correct": correct,
               "accuracy": round(correct / len(scored), 4) if scored else None,
               "strict_correct": sum(r["correct"] and r["well_formed"] for r in scored),
               "invalid_labels": sum(not r["valid_label"] for r in scored),
               "malformed": sum(not r["well_formed"] for r in scored),
               "empty_retries": sum(r.get("empty_retries", 0) for r in results),
               "cache_hits": sum(r["cache"] == "hit" for r in scored),
               "cost_usd": round(sum(r["cost_usd"] for r in scored), 6),
               "avg_tokens": round(sum(r["tokens"] or 0 for r in scored) / max(len(scored), 1)),
               "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tag = f"{args.split}-{args.alias}-{args.model}-r{args.repeat}"
    json.dump({"summary": summary, "rows": results},
              open(os.path.join(HERE, "results", f"eval-{tag}.json"), "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=setup)
    d = sub.add_parser("dataset")
    d.add_argument("--round", type=int, default=1)
    d.set_defaults(fn=dataset)
    o = sub.add_parser("optimize")
    o.add_argument("--dataset", required=True)
    o.add_argument("--model", required=True)
    o.add_argument("--optimizer", required=True)
    o.add_argument("--judge", required=True)
    o.add_argument("--drafts", type=int, default=3)
    o.add_argument("--baseline", default="lazy", help="alias the rewrites must beat")
    o.set_defaults(fn=optimize)
    rp = sub.add_parser("report")
    rp.add_argument("--run", required=True)
    rp.set_defaults(fn=report)
    pr = sub.add_parser("promote")
    pr.add_argument("--run", required=True)
    pr.add_argument("--candidate", required=True)
    pr.add_argument("--alias", default="fixed")
    pr.set_defaults(fn=promote)
    e = sub.add_parser("eval")
    e.add_argument("--alias", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--repeat", type=int, default=1)
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--show", action="store_true", help="print each message as it is scored")
    e.set_defaults(fn=eval_split)
    a = p.parse_args()
    a.fn(a)
