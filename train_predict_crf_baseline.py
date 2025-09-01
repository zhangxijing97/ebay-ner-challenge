"""
train_predict_crf_baseline.py

Train a per-category CRF (Conditional Random Fields) token classifier on
Tagged_Titles_Train_train.tsv, then predict tags for Tagged_Titles_TitlesOnly.tsv.

Input:
    - Tagged_Titles_Train_train.tsv  (token-level training data)
    - Tagged_Titles_TitlesOnly.tsv   (titles only, for prediction)
    - category2aspects.json          (allowed tag set per category)

Output:
    - model_output/predictions_token_level_crf.tsv
      (token-level predictions ready for evaluator)
"""

import pandas as pd, json, re
from pathlib import Path
from collections import defaultdict
import sklearn_crfsuite

# ---------- paths ----------
train_path   = Path("processed_data/Tagged_Titles_Train_train.tsv")
titles_path  = Path("model_input/Tagged_Titles_TitlesOnly.tsv")
cat_json     = Path("processed_data/category2aspects.json")
out_dir      = Path("model_output"); out_dir.mkdir(parents=True, exist_ok=True)
pred_path    = out_dir / "predictions_token_level_crf.tsv"

# ---------- load ----------
train  = pd.read_csv(train_path, sep="\t", dtype=str, keep_default_na=False, engine="python")
titles = pd.read_csv(titles_path, sep="\t", dtype=str, keep_default_na=False, engine="python")
cat2aspects = json.loads(cat_json.read_text(encoding="utf-8"))

# ---------- tokenizer ----------
tok_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+|[+\-–—/(),.:;_#]|Ø|ø|°|mm|MM", re.UNICODE)
def tokenize(title: str):
    return [m.group(0) for m in tok_re.finditer(str(title))]

# ---------- feature builder ----------
def word_shape(s: str):
    out = []
    for ch in s:
        if ch.isupper(): out.append("A")
        elif ch.islower(): out.append("a")
        elif ch.isdigit(): out.append("9")
        else: out.append("-")
    comp, prev = [], None
    for c in out:
        if c != prev: comp.append(c); prev = c
    return "".join(comp)

def token_features(seq, i):
    w = seq[i]; wl = w.lower()
    feats = {
        "bias": 1.0,
        "word.lower": wl,
        "word.isupper": w.isupper(),
        "word.isdigit": w.isdigit(),
        "shape": word_shape(w),
        "prefix2": wl[:2], "prefix3": wl[:3],
        "suffix2": wl[-2:], "suffix3": wl[-3:],
    }
    if i > 0:
        w0 = seq[i-1].lower()
        feats.update({"-1:word.lower": w0})
    else:
        feats["BOS"] = True
    if i < len(seq)-1:
        w1 = seq[i+1].lower()
        feats.update({"+1:word.lower": w1})
    else:
        feats["EOS"] = True
    return feats

def sent2features(sent): return [token_features(sent, i) for i in range(len(sent))]

# ---------- prepare training ----------
cat_models = {}
cat_allowed = {}

for cat, g in train.groupby("Category", sort=False):
    allowed = set(cat2aspects.get(str(cat), [])) | {"O"}
    cat_allowed[cat] = allowed

    # group by (record, title) to keep sequences
    X_seq, y_seq = [], []
    for (rn, title), gg in g.groupby(["Record Number","Title"], sort=False):
        toks = gg["Token"].astype(str).tolist()
        tags = [t if t in allowed and t != "" else "O" for t in gg["Tag"].astype(str).tolist()]
        X_seq.append(sent2features(toks))
        y_seq.append(tags)

    if not X_seq: continue

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", 
        max_iterations=100,
        all_possible_transitions=True,
        c1=0.1, c2=0.1
    )
    crf.fit(X_seq, y_seq)
    cat_models[cat] = crf

print("Trained CRF models for categories:", list(cat_models.keys()))

# ---------- predict ----------
pred_rows = []
for _, r in titles.iterrows():
    rn, cat, title = r["Record Number"], r["Category"], r["Title"]
    toks = tokenize(title)
    allowed = cat_allowed.get(cat, {"O"})
    crf = cat_models.get(cat, None)

    if crf is None or not toks:
        for t in toks:
            pred_rows.append([rn, cat, title, t, "O"])
        continue

    X = sent2features(toks)
    y_pred = crf.predict_single(X)
    # restrict to allowed labels
    y_pred = [y if y in allowed else "O" for y in y_pred]

    pred_rows.extend([[rn, cat, title, t, y] for t,y in zip(toks,y_pred)])

pred = pd.DataFrame(pred_rows, columns=["Record Number","Category","Title","Token","Tag"])
pred.to_csv(pred_path, sep="\t", index=False, encoding="utf-8")
print(f"Saved predictions -> {pred_path.resolve()}")