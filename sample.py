#!/usr/bin/env python3
"""Draw the PromptSloth splits from Banking77 (PolyAI, CC-BY-4.0).

train split -> the optimizer's dataset: TRAIN_PER_INTENT messages per intent.
test split  -> the held-out set the optimizer never sees: TEST_PER_INTENT per intent.
Seed 42, so every number in the article regenerates.
"""
import csv, json, os, random, urllib.request

SRC = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PER_INTENT, TEST_PER_INTENT = 2, 4


def load(split):
    path = os.path.join(HERE, "data", f"{split}.csv")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(SRC.format(split), path)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified(rows, per_intent, rng):
    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r["text"])
    out = []
    for intent in sorted(by):
        for text in rng.sample(by[intent], per_intent):
            out.append({"text": text, "intent": intent})
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    rng = random.Random(42)
    train, test = stratified(load("train"), TRAIN_PER_INTENT, rng), stratified(load("test"), TEST_PER_INTENT, rng)
    intents = sorted({r["intent"] for r in train})
    json.dump({"intents": intents, "train": train, "test": test},
              open(os.path.join(HERE, "data", "splits.json"), "w"), indent=1)
    print(f"{len(intents)} intents, {len(train)} train, {len(test)} test")
