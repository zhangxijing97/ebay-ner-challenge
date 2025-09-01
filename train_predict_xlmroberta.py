"""
train_predict_xlmroberta.py

Fine-tune XLM-RoBERTa for token classification per Category on
Tagged_Titles_Train_train.tsv, then predict tags for Tagged_Titles_TitlesOnly.tsv.
Outputs token-level predictions TSV compatible with your evaluator.

Inputs:
  - Tagged_Titles_Train_train.tsv      (token-level: Record Number, Category, Title, Token, Tag)
  - Tagged_Titles_TitlesOnly.tsv       (titles only:  Record Number, Category, Title)
  - category2aspects.json              (allowed tag set per category)

Outputs:
  - model_output/predictions_token_level_xlmroberta.tsv
  - models/xlmr_cat_<cat>/             (one folder per category model)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid fork deadlocks & noisy warnings

import json, re
from pathlib import Path
from typing import List, Dict
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments
)
from datasets import Dataset

# --------------------- device policy (MOST STABLE) ---------------------
USE_CUDA = torch.cuda.is_available()
# Intentionally disable MPS to avoid "Placeholder storage..." crash on Apple GPUs.
USE_MPS = False
print(f"[Device] CUDA={USE_CUDA} | MPS={USE_MPS} (disabled)")

# --------------------- paths ---------------------
train_path  = Path("processed_data/Tagged_Titles_Train_train.tsv")
titles_path = Path("model_input/Tagged_Titles_TitlesOnly.tsv")
cat_json    = Path("processed_data/category2aspects.json")
out_dir     = Path("model_output"); out_dir.mkdir(parents=True, exist_ok=True)
model_root  = Path("models"); model_root.mkdir(parents=True, exist_ok=True)
pred_path   = out_dir / "predictions_token_level_xlmroberta.tsv"

# --------------------- hyperparams ---------------------
MODEL_NAME   = "xlm-roberta-base"
EPOCHS       = 3
LR           = 2e-5
BATCH_SIZE   = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED         = 42

# --------------------- load data ---------------------
train_df  = pd.read_csv(train_path,  sep="\t", dtype=str, keep_default_na=False, engine="python")
titles_df = pd.read_csv(titles_path, sep="\t", dtype=str, keep_default_na=False, engine="python")
cat2aspects: Dict[str, List[str]] = json.loads(cat_json.read_text(encoding="utf-8"))

# --------------------- tokenizer & tokenization ---------------------
# Keep symbols similar to training tokens
tok_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+|[+\-–—/(),.:;_#]|Ø|ø|°|mm|MM", re.UNICODE)
def tokenize_title(title: str) -> List[str]:
    return [m.group(0) for m in tok_re.finditer(str(title))]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def encode_example(words: List[str], tags: List[str], label2id: Dict[str, int]):
    """Encode pre-tokenized words, align labels to first subword; others -> -100."""
    enc = tokenizer(words, is_split_into_words=True, truncation=True, return_offsets_mapping=False)
    word_ids = enc.word_ids()
    labels = []
    prev_wi = None
    for wi in word_ids:
        if wi is None:
            labels.append(-100)
        else:
            if wi != prev_wi:
                lab = label2id[tags[wi]] if wi < len(tags) else label2id["O"]
                labels.append(lab)
                prev_wi = wi
            else:
                labels.append(-100)  # subsequent subwords ignored in loss
    enc["labels"] = labels
    return enc

def build_dataset_for_category(cat: str):
    """Return HF Dataset + label maps for one category."""
    g = train_df[train_df["Category"] == cat]
    allowed = set(cat2aspects.get(str(cat), [])) | {"O"}

    sents = []
    for (_, title), gg in g.groupby(["Record Number", "Title"], sort=False):
        words = gg["Token"].astype(str).tolist()
        tags  = [t if t in allowed and t != "" else "O" for t in gg["Tag"].astype(str).tolist()]
        sents.append((words, tags))

    labels_sorted = sorted(allowed)
    label2id = {lbl: i for i, lbl in enumerate(labels_sorted)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    enc_list = [encode_example(words, tags, label2id) for words, tags in sents]
    ds = Dataset.from_list(enc_list)
    return ds, label2id, id2label, labels_sorted

def train_one_category(cat: str):
    ds, label2id, id2label, labels_sorted = build_dataset_for_category(cat)
    if len(ds) == 0:
        return None, label2id, id2label, labels_sorted

    model_path = model_root / f"xlmr_cat_{cat}"
    model_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels_sorted), id2label=id2label, label2id=label2id
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=str(model_path),
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        save_strategy="epoch",
        seed=SEED,
        report_to="none",

        # Stability settings:
        no_cuda=not USE_CUDA,          # use CUDA if available, else CPU
        use_mps_device=False,          # disable MPS explicitly
        fp16=False, bf16=False,        # no mixed precision; keep it simple & stable
        dataloader_pin_memory=False,   # avoids irrelevant pin_memory warnings on macOS
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    return model_path, label2id, id2label, labels_sorted

# --------------------- train per category ---------------------
cats = sorted(train_df["Category"].unique(), key=lambda x: (str(x).isdigit(), x))
trained_meta = {}  # cat -> (model_path, id2label, allowed)
for cat in cats:
    mp, l2i, i2l, labels_sorted = train_one_category(cat)
    if mp is not None:
        trained_meta[cat] = (mp, i2l, set(labels_sorted))
print("[Train] Finished. Trained categories:", list(trained_meta.keys()))

# --------------------- inference ---------------------
def predict_words(model_path: Path, words: List[str], id2label, allowed: set):
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # keep off MPS
    model.to(device)
    with torch.no_grad():
        enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[0]  # [seq_len, num_labels]
        word_ids = tokenizer(words, is_split_into_words=True).word_ids()

        # take label at first subword for each word
        preds, seen = [], set()
        for i, wi in enumerate(word_ids):
            if wi is None: 
                continue
            if wi in seen:
                continue
            seen.add(wi)
            lab_id = int(torch.argmax(logits[i]).item())
            lab = id2label.get(lab_id, "O")
            preds.append(lab if lab in allowed else "O")
        return preds

pred_rows = []
for _, r in titles_df.iterrows():
    rn, cat, title = r["Record Number"], r["Category"], r["Title"]
    tokens = tokenize_title(title)
    meta = trained_meta.get(cat)
    if meta is None or not tokens:
        for t in tokens:
            pred_rows.append([rn, cat, title, t, "O"])
        continue
    model_path, id2label, allowed = meta
    y = predict_words(model_path, tokens, id2label, allowed)
    # align lengths (rarely needed if truncation happened)
    if len(y) != len(tokens):
        y = (y[:len(tokens)] + ["O"] * max(0, len(tokens) - len(y)))
    pred_rows.extend([[rn, cat, title, t, tag] for t, tag in zip(tokens, y)])

pred_df = pd.DataFrame(pred_rows, columns=["Record Number","Category","Title","Token","Tag"])
pred_df.to_csv(pred_path, sep="\t", index=False, encoding="utf-8")
print(f"[Predict] Saved -> {pred_path.resolve()}")

# --------------------- optional: quick eval (if your evaluator is present) ---------------------
try:
    # If your notebook defines these helpers, this will print scores immediately.
    from pathlib import Path as _P
    gold_path = _P("processed_data/Tagged_Titles_Train_val.tsv")
    if gold_path.exists():
        # Expect your previously-defined helpers in this runtime; ignore if absent.
        from __main__ import load_token_tsv, evaluate_token_files  # noqa
        gold_tok = load_token_tsv(gold_path)
        pred_tok = load_token_tsv(pred_path)
        final_beta, _, _ = evaluate_token_files(gold_tok, pred_tok, beta=0.2)
        final_f1,   _, _ = evaluate_token_files(gold_tok, pred_tok, beta=1.0)
        print(f"[Eval] Fβ(0.2): {final_beta:.6f} | F1: {final_f1:.6f}")
except Exception as e:
    print(f"[Eval] Skipped quick eval ({e})")