# Level-1: Per-token linear classifier (one model per Category)
# Produces predictions in token-level TSV ready for your evaluator.

import pandas as pd, json, re, numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

# ---------- paths ----------
train_path   = Path("processed_data/Tagged_Titles_Train_train.tsv")
titles_path  = Path("model_input/Tagged_Titles_TitlesOnly.tsv")
cat_json     = Path("processed_data/category2aspects.json")
out_dir      = Path("model_output"); out_dir.mkdir(parents=True, exist_ok=True)
pred_path    = out_dir / "predictions_token_logreg.tsv"

# ---------- I/O ----------
train = pd.read_csv(train_path, sep="\t", dtype=str, keep_default_na=False, engine="python")
titles = pd.read_csv(titles_path, sep="\t", dtype=str, keep_default_na=False, engine="python")
cat2aspects = json.loads(cat_json.read_text(encoding="utf-8"))

# ---------- tokenization ----------
# Keep words and symbols similar to training (digits, letters, '+', 'Ø', '-', '/', ',', '()')
tok_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+|[+\-–—/(),.:;_#]|Ø|ø|°|mm|MM", re.UNICODE)

def tokenize(title: str):
    return [m.group(0) for m in tok_re.finditer(str(title))]

# ---------- feature functions ----------
def word_shape(s: str):
    # Map chars to categories: upper->A, lower->a, digit->9, other->-
    out = []
    for ch in s:
        if ch.isupper(): out.append("A")
        elif ch.islower(): out.append("a")
        elif ch.isdigit(): out.append("9")
        else: out.append("-")
    # collapse repeats (e.g., Aaaa99 -> Aa9)
    comp, prev = [], None
    for c in out:
        if c != prev: comp.append(c); prev = c
    return "".join(comp)

def gazetteers():
    # Ultra-light lists;可逐步扩充
    brands = {"ZIMMERMANN","BOSCH","ATE","SKF","TRW","CONTINENTAL","FEBI","VAICO","EBC","NK","TEXTAR"}
    pos_set = {"VORNE","HINTEN","VA","HA"}
    parts = {"BREMSSCHEIBE","BREMSSCHEIBEN","BREMSBELÄGE","ZAHNRIEMENSATZ","WASSERPUMPE","KEILRIPPENRIEMEN"}
    return brands, pos_set, parts

BRANDS, POS_SET, PARTS = gazetteers()

def feats(seq, i):
    """Contextual features for token at position i."""
    w = seq[i]
    wl = w.lower()
    f = {
        "w": w,
        "wl": wl,
        "is_upper": w.isupper(),
        "is_lower": w.islower(),
        "is_digit": w.isdigit(),
        "shape": word_shape(w),
        "pref1": wl[:1], "pref2": wl[:2], "pref3": wl[:3], "pref4": wl[:4],
        "suf1": wl[-1:], "suf2": wl[-2:], "suf3": wl[-3:], "suf4": wl[-4:],
        "in_brand": w.upper() in BRANDS,
        "in_pos": w.upper() in POS_SET,
        "in_parts": w.upper() in PARTS,
        "has_digit": any(ch.isdigit() for ch in w),
        "has_alpha": any(ch.isalpha() for ch in w),
        "len": min(15, len(w)),
    }
    # context window ±2
    for off in (-2, -1, 1, 2):
        j = i + off
        key = f"ctx{off}"
        if 0 <= j < len(seq):
            ww, wwl = seq[j], seq[j].lower()
            f.update({
                f"{key}_w": ww,
                f"{key}_wl": wwl,
                f"{key}_shape": word_shape(ww),
                f"{key}_is_digit": ww.isdigit(),
            })
        else:
            f[f"{key}_w"] = "<BOS/EOS>"
    return f

# ---------- prepare training per category ----------
# Keep Tag set = cat2aspects[cat] ∪ {"O"}
cat_models = {}              # cat -> (DictVectorizer, LogisticRegression)
cat_label_set = {}           # cat -> allowed labels

for cat, g in train.groupby("Category", sort=False):
    allowed = set(cat2aspects.get(str(cat), [])) | {"O"}
    cat_label_set[cat] = sorted(allowed)

    # build sequences by title (to get context), but sample per token
    rows = []
    for (rn, title), gg in g.groupby(["Record Number","Title"], sort=False):
        toks = list(gg["Token"].astype(str).tolist())
        tags = list(gg["Tag"].astype(str).tolist())
        for i, (t, tag) in enumerate(zip(toks, tags)):
            tag = tag if tag in allowed and tag != "" else "O"
            rows.append((feats(toks, i), tag))

    if not rows:
        continue

    X_dict = [r[0] for r in rows]
    y = [r[1] for r in rows]

    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(X_dict)

    # class weights to fight O-dominance (auto balance)
    clf = LogisticRegression(
        solver="saga", max_iter=200, n_jobs=-1, class_weight="balanced",
        multi_class="auto", C=2.0, verbose=0
    )
    clf.fit(X, y)

    cat_models[cat] = (dv, clf)

print(f"Trained models for categories: {sorted(cat_models.keys())}")

# ---------- inference on TitlesOnly ----------
pred_rows = []
for _, r in titles.iterrows():
    rn, cat, title = r["Record Number"], r["Category"], r["Title"]
    allowed = cat_label_set.get(cat, {"O"})
    dv_clf = cat_models.get(cat, None)

    toks = tokenize(title)
    if dv_clf is None:
        # no model for this category -> all 'O'
        for t in toks:
            pred_rows.append([rn, cat, title, t, "O"])
        continue

    dv, clf = dv_clf
    X_dict = [feats(toks, i) for i in range(len(toks))]
    X = dv.transform(X_dict)
    y_pred = clf.predict(X)

    # safety: restrict to allowed labels
    y_pred = [yp if yp in allowed else "O" for yp in y_pred]
    pred_rows.extend([[rn, cat, title, t, yp] for t, yp in zip(toks, y_pred)])

pred = pd.DataFrame(pred_rows, columns=["Record Number","Category","Title","Token","Tag"])
pred.to_csv(pred_path, sep="\t", index=False, encoding="utf-8")
print(f"Saved predictions -> {pred_path.resolve()}")