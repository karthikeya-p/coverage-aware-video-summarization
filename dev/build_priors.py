"""Build weights/verb_labels.json, weights/noun_labels.json (ActionFormer label_id -> EPIC class key) and
weights/verb_noun_cooc.json (counts of (verb_id, noun_id) pairs in the EPIC-100 *train* split only).

Label ids: taken from the public ActionFormer annotation json when present (authoritative), otherwise from
EPIC_100_{verb,noun}_classes.csv assuming label_id == EPIC class id (what ActionFormer did). The two are
cross-checked when both are available.
"""
import csv, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))  # inference code
import config as C

ANN = os.path.join(C.WEIGHTS, "epic-kitchens-100-annotations")
PUB = os.path.join(C.WEIGHTS, "epic_kitchens_public")


def from_csv(kind):
    p = os.path.join(ANN, f"EPIC_100_{kind}_classes.csv")
    if not os.path.exists(p):
        return None
    with open(p, newline="") as f:
        return {int(r["id"]): r["key"] for r in csv.DictReader(f)}


def from_public_json(kind):
    hits = glob.glob(os.path.join(PUB, "**", f"*{kind}*.json"), recursive=True)
    if not hits:
        return None
    db = json.load(open(hits[0]))
    db = db.get("database", db)
    m = {}
    for v in db.values():
        for a in v.get("annotations", []):
            m[int(a["label_id"])] = a["label"]
    return m


def main():
    for kind, out in (("verb", C.VERB_LABELS), ("noun", C.NOUN_LABELS)):
        pub, csvm = from_public_json(kind), from_csv(kind)
        if pub and csvm:
            diff = [i for i in pub if i not in csvm]
            print(f"[{kind}] public json vs class csv: {len(pub)} ids, {len(diff)} ids missing from the csv"
                  + (f" e.g. {diff[:5]}" if diff else ""))
        # The public json stores the class id as its "label" string, so names must come from the EPIC class csv;
        # the public json only confirms that ActionFormer label_id == EPIC class id.
        m = csvm or pub
        vocab_path = C.SUPPORTED_CLASSES
        if m and os.path.exists(vocab_path):
            allowed = set(json.load(open(vocab_path))[kind + "s"])
            missing = sorted(v for v in m.values() if v not in allowed)
            print(f"[{kind}] {len(m) - len(missing)}/{len(m)} class names are in supported_classes.json"
                  + (f"; NOT supported e.g. {missing[:8]}" if missing else ""))
        if not m:
            print(f"!! no label source for {kind}; run setup.sh weights step"); continue
        # public json labels are EPIC keys already (e.g. 'board:chopping'); keep as-is
        json.dump({str(k): v for k, v in sorted(m.items())}, open(out, "w"), indent=0)
        print(f"wrote {out} ({len(m)} classes)")

    train = os.path.join(ANN, "EPIC_100_train.csv")
    if os.path.exists(train):
        cooc = {}
        with open(train, newline="") as f:
            for r in csv.DictReader(f):
                k = f"{int(r['verb_class'])},{int(r['noun_class'])}"
                cooc[k] = cooc.get(k, 0) + 1
        json.dump(cooc, open(C.COOC_PRIOR, "w"))
        print(f"wrote {C.COOC_PRIOR}: {len(cooc)} distinct verb-noun pairs seen in TRAIN (val never used)")
    else:
        print("!! EPIC_100_train.csv not found; co-occurrence prior disabled (prior=1)")


if __name__ == "__main__":
    main()
