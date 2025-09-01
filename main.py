# # main.py
# import json
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# import torch
# from transformers import AutoTokenizer, AutoModel

# from neural_network import NeuralNetwork

# # ------------------ Step 0: 配置 ------------------
# print("Step 0: 配置路径...")
# MODEL_NAME = "xlm-roberta-base"
# BIO_PATH = Path("cache/train_bio.parquet")      # record-level：tokens(list[str]), tags(list[str])
# LABEL_PATH = Path("cache/label_set.json")       # 你的 44 类标签（含 'O'）

# # ------------------ Step 1: 读取数据/标签 ------------------
# print("Step 1: 读取数据和标签...")
# df = pd.read_parquet(BIO_PATH)
# with open(LABEL_PATH, "r", encoding="utf-8") as f:
#     label_list = json.load(f)
# label2id = {l: i for i, l in enumerate(label_list)}
# if "O" not in label2id:
#     label_list.append("O")
#     label2id["O"] = len(label_list) - 1
# num_labels = len(label_list)
# print(f"数据量(行): {len(df)}, 标签数: {num_labels}")

# # ------------------ Step 2: 加载 tokenizer/encoder ------------------
# print("Step 2: 加载 tokenizer 和 encoder...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# encoder = AutoModel.from_pretrained(MODEL_NAME)
# encoder.eval(); encoder.requires_grad_(False)
# print("加载完毕")

# # ------------------ 工具：tokens/tags 规范化 ------------------
# def coerce_tokens(tokens):
#     import numpy as np
#     if isinstance(tokens, str):
#         tokens = tokens.split()
#     if isinstance(tokens, (pd.Series, np.ndarray)):
#         tokens = tokens.tolist()
#     if not isinstance(tokens, (list, tuple)):
#         raise TypeError(f"tokens must be list-like or str, got {type(tokens)}")
#     cleaned = []
#     for t in tokens:
#         if t is None: continue
#         if isinstance(t, float) and np.isnan(t): continue
#         t = str(t).strip()
#         if t == "": continue
#         cleaned.append(t)
#     return cleaned

# def coerce_tags(tags):
#     import numpy as np
#     if isinstance(tags, str):
#         tags = [p.strip() for p in tags.split()]
#     if isinstance(tags, (pd.Series, np.ndarray)):
#         tags = tags.tolist()
#     if not isinstance(tags, (list, tuple)):
#         raise TypeError(f"tags must be list-like or str, got {type(tags)}")
#     out = []
#     for t in tags:
#         if t is None: out.append("O"); continue
#         if isinstance(t, float) and np.isnan(t): out.append("O"); continue
#         t = str(t).strip()
#         out.append(t if t != "" else "O")
#     return out

# # ------------------ Step 3: 定义向量提取 ------------------
# print("Step 3: 定义向量提取函数...")
# def title_to_word_embeds_and_count(tokens):
#     """
#     返回:
#       word_vecs: np.ndarray [n_words_extracted, H]
#       n_words_extracted: 实际抽到的词数（考虑 512 subword 截断）
#     """
#     tokens = coerce_tokens(tokens)
#     if len(tokens) == 0:
#         return None, 0

#     enc = tokenizer(
#         tokens,
#         is_split_into_words=True,
#         return_tensors="pt",
#         truncation=True,
#         padding=False
#     )
#     with torch.no_grad():
#         out = encoder(
#             input_ids=enc["input_ids"],
#             attention_mask=enc.get("attention_mask", None)
#         )
#         hs = out.last_hidden_state.squeeze(0)  # [T_sub, H]

#     word_ids = enc.word_ids(0)  # 长度 T_sub
#     seen_word_ids = [wid for wid in word_ids if wid is not None]
#     if len(seen_word_ids) == 0:
#         return None, 0
#     n_words_extracted = max(seen_word_ids) + 1

#     # 聚合同一 word 的 subwords（均值）
#     word_vecs, cur = [], []
#     prev_wid = None
#     for idx, wid in enumerate(word_ids):
#         if wid is None:
#             continue
#         if (prev_wid is None) or (wid == prev_wid):
#             cur.append(hs[idx].cpu().numpy())
#         else:
#             word_vecs.append(np.mean(cur, axis=0))
#             cur = [hs[idx].cpu().numpy()]
#         prev_wid = wid
#     if cur:
#         word_vecs.append(np.mean(cur, axis=0))

#     if len(word_vecs) == 0:
#         return None, 0
#     if len(word_vecs) > n_words_extracted:
#         word_vecs = word_vecs[:n_words_extracted]

#     return np.stack(word_vecs, axis=0), n_words_extracted

# print("Step 3: 函数定义完成")

# # ------------------ Step 4: 按 record 级划分 ------------------
# print("Step 4: 按 record_id 划分训练/验证...")
# if "record_id" not in df.columns and "Record Number" in df.columns:
#     df = df.rename(columns={"Record Number": "record_id"})
# record_ids = pd.unique(df["record_id"])
# train_ids, valid_ids = train_test_split(record_ids, test_size=0.2, random_state=42)
# train_df = df[df["record_id"].isin(train_ids)].reset_index(drop=True)
# valid_df = df[df["record_id"].isin(valid_ids)].reset_index(drop=True)
# print(f"train_df 行数: {len(train_df)}, valid_df 行数: {len(valid_df)}")

# # ------------------ Step 5: 抽取函数（对一个子集） ------------------
# def extract_Xy(df_subset, tag_map, enc_hidden_size):
#     Xs, ys = [], []
#     stats = {
#         "empty_tokens": 0,
#         "bad_type_tokens": 0,
#         "bad_type_tags": 0,
#         "mismatch_len_before": 0,
#         "truncated_align_fixed": 0,
#         "still_mismatch_after": 0,
#         "empty_after_encode": 0,
#         "ok_rows": 0
#     }
#     for i, row in df_subset.iterrows():
#         try:
#             tokens, tags = row.get("tokens", None), row.get("tags", None)
#             try:
#                 tokens = coerce_tokens(tokens)
#             except Exception:
#                 stats["bad_type_tokens"] += 1
#                 continue
#             try:
#                 tags = coerce_tags(tags)
#             except Exception:
#                 stats["bad_type_tags"] += 1
#                 continue

#             if len(tokens) == 0:
#                 stats["empty_tokens"] += 1
#                 continue

#             if len(tokens) != len(tags):
#                 stats["mismatch_len_before"] += 1

#             emb, n_words = title_to_word_embeds_and_count(tokens)
#             if emb is None or n_words == 0:
#                 stats["empty_after_encode"] += 1
#                 continue

#             if len(tokens) != n_words:
#                 tokens = tokens[:n_words]
#             if len(tags) != n_words:
#                 tags = tags[:n_words]
#                 stats["truncated_align_fixed"] += 1

#             if len(tokens) != len(tags) or emb.shape[0] != len(tags):
#                 stats["still_mismatch_after"] += 1
#                 continue

#             y_ids = np.array([tag_map.get(t, tag_map["O"]) for t in tags], dtype=np.int64)
#             Xs.append(emb); ys.append(y_ids)
#             stats["ok_rows"] += 1

#             if (i + 1) % 500 == 0:
#                 print(f"  子集已处理 {i+1} 行... (OK累计: {stats['ok_rows']})")
#         except Exception as e:
#             rid = row.get("record_id", "?")
#             print(f"  跳过子集第 {i} 行（record_id={rid}）: {e}")

#     if Xs:
#         X = np.vstack(Xs)
#         y = np.hstack(ys)
#     else:
#         X = np.zeros((0, enc_hidden_size))
#         y = np.zeros((0,), dtype=np.int64)
#     return X, y, stats

# # ------------------ Step 6: 分别抽取 Train/Valid 的 (X, y) ------------------
# print("Step 6: 从训练子集提取向量...")
# enc_hidden = encoder.config.hidden_size
# X_train_raw, y_train, stats_train = extract_Xy(train_df, label2id, enc_hidden)
# print("Train 诊断:", stats_train, f"X_train_raw.shape={X_train_raw.shape}, y_train.shape={y_train.shape}")

# print("Step 6b: 从验证子集提取向量...")
# X_valid_raw, y_valid, stats_valid = extract_Xy(valid_df, label2id, enc_hidden)
# print("Valid 诊断:", stats_valid, f"X_valid_raw.shape={X_valid_raw.shape}, y_valid.shape={y_valid.shape}")

# if X_train_raw.shape[0] == 0 or X_valid_raw.shape[0] == 0:
#     raise RuntimeError("训练或验证子集中没有可用样本，请检查诊断统计。")

# # ------------------ Step 7: 标准化（只用训练集拟合） ------------------
# print("Step 7: 标准化（fit on train, transform both）...")
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train_raw)
# X_valid = scaler.transform(X_valid_raw)
# print("标准化完成")

# # ------------------ Step 8: one-hot 标签 ------------------
# print("Step 8: one-hot 编码标签...")
# def to_one_hot(y_, C):
#     oh = np.zeros((y_.shape[0]), dtype=np.int64)  # 仅用于占位，真实 one-hot 在 NN 内部梯度使用
#     # 这里仍然需要真正 one-hot 以配合我们的 NN：
#     oh2 = np.zeros((y_.shape[0], C), dtype=np.float32)
#     oh2[np.arange(y_.shape[0]), y_] = 1.0
#     return oh2

# Y_train = to_one_hot(y_train, num_labels)
# Y_valid = to_one_hot(y_valid, num_labels)
# print("编码完成")

# # ------------------ Step 9: 定义并训练 DNN ------------------
# print("Step 9: 定义并训练 DNN...")
# input_size = X_train.shape[1]
# layers = [
#     (input_size, 'ReLU'),
#     (256, 'ReLU'),
#     (num_labels, 'Softmax')
# ]
# epochs = 2000
# learning_rate = 0.05

# nn = NeuralNetwork(layers)
# nn.train(X_train, Y_train, epochs, learning_rate)
# print("训练完成")

# # ------------------ Step 10: 预测 ------------------
# print("Step 10: 验证集预测...")
# probs = nn.forward(X_valid)[f'A{len(layers)-1}']   # [N, C]
# pred = np.argmax(probs, axis=1)
# print("预测完成")

# # ------------------ Step 11: 简单 token 级评估 ------------------
# print("Step 11: 计算指标...")
# def precision_recall(y_true, y_pred):
#     TP = np.sum(y_true == y_pred)
#     FP = np.sum(y_true != y_pred)
#     precision = TP / (TP + FP + 1e-12)
#     recall = TP / (len(y_true) + 1e-12)
#     return precision, recall

# def f_beta(p, r, beta=0.2):
#     if p == 0 and r == 0: return 0.0
#     b2 = beta * beta
#     return (1+b2)*p*r / (b2*p + r + 1e-12)

# p, r = precision_recall(y_valid, pred)
# print(f"最终结果（record 级切分）→ Precision={p:.4f}, Recall={r:.4f}, F0.2={f_beta(p,r,0.2):.4f}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build fixed-length token-classification datasets from aligned TSVs.

Input  (per row = a subword):
  - Record Number | Subword | WordID | BIO_Label | Label_ID | Input_ID
Output (per sequence = one Record Number):
  - input_ids [N, max_len]
  - attention_mask [N, max_len]
  - labels [N, max_len]  (pad = -100)
"""

# import argparse
# from pathlib import Path
# import pandas as pd
# import torch
# from transformers import AutoTokenizer

# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train_tsv", default="processed_data/aligned_train_subwords.tsv")
#     ap.add_argument("--val_tsv",   default="processed_data/aligned_val_subwords.tsv")
#     ap.add_argument("--out_dir",   default="processed_data")
#     ap.add_argument("--model_name", default="xlm-roberta-base",
#                     help="Used only to get pad_token_id safely")
#     ap.add_argument("--max_len", type=int, default=160)
#     return ap.parse_args()

# def load_and_group(tsv_path: Path):
#     df = pd.read_csv(tsv_path, sep="\t")
#     need = {"Record Number", "Input_ID", "Label_ID"}
#     if not need.issubset(df.columns):
#         missing = need - set(df.columns)
#         raise ValueError(f"Missing columns in {tsv_path}: {missing}")
#     # keep order within each Record Number as in file (sort=False)
#     groups = [(rid, g["Input_ID"].tolist(), g["Label_ID"].tolist())
#               for rid, g in df.groupby("Record Number", sort=False)]
#     return groups

# def pad_truncate(seq_ids, seq_lbls, max_len, pad_id, ignore_index=-100):
#     # truncate (keep left-to-right; titles通常短，直接右裁剪)
#     ids  = seq_ids[:max_len]
#     lbls = seq_lbls[:max_len]
#     # pad to max_len
#     if len(ids) < max_len:
#         pad_n = max_len - len(ids)
#         ids  = ids + [pad_id] * pad_n
#         lbls = lbls + [ignore_index] * pad_n
#     # attention mask: 1 for non-pad, 0 for pad
#     attn = [0 if tok == pad_id else 1 for tok in ids]
#     return ids, attn, lbls

# def build_tensor_pack(groups, max_len, pad_id):
#     input_ids, attention_mask, labels, rids = [], [], [], []
#     for rid, ids, lbls in groups:
#         ids_i, att_i, lab_i = pad_truncate(ids, lbls, max_len, pad_id, ignore_index=-100)
#         input_ids.append(ids_i)
#         attention_mask.append(att_i)
#         labels.append(lab_i)
#         rids.append(int(rid))
#     pack = {
#         "input_ids":      torch.tensor(input_ids, dtype=torch.long),
#         "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
#         "labels":         torch.tensor(labels, dtype=torch.long),
#         "rids":           torch.tensor(rids, dtype=torch.long),
#         "pad_id":         int(pad_id),
#         "max_len":        int(max_len),
#     }
#     return pack

# def main():
#     args = parse_args()
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Use tokenizer only to obtain a correct pad_token_id for the model vocab
#     tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
#     pad_id = tok.pad_token_id
#     if pad_id is None:
#         # XLM-R uses 1 as <pad>, but fall back safely
#         pad_id = 1

#     # Train
#     train_groups = load_and_group(Path(args.train_tsv))
#     train_pack = build_tensor_pack(train_groups, args.max_len, pad_id)
#     torch.save(train_pack, out_dir / "train_dataset.pt")
#     print(f"[train] sequences: {len(train_pack['rids'])} | "
#           f"shape: {tuple(train_pack['input_ids'].shape)} "
#           f"| saved -> {out_dir/'train_dataset.pt'}")

#     # Val (optional but default provided)
#     val_path = Path(args.val_tsv)
#     if val_path.exists():
#         val_groups = load_and_group(val_path)
#         val_pack = build_tensor_pack(val_groups, args.max_len, pad_id)
#         torch.save(val_pack, out_dir / "val_dataset.pt")
#         print(f"[val]   sequences: {len(val_pack['rids'])} | "
#               f"shape: {tuple(val_pack['input_ids'].shape)} "
#               f"| saved -> {out_dir/'val_dataset.pt'}")
#     else:
#         print(f"[val] skipped (file not found): {val_path}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train & evaluate a token-classification model (XLM-R) on aligned subword datasets.

Inputs (produced by 05_build_datasets.py):
  processed_data/train_dataset.pt  # dict: input_ids, attention_mask, labels, rids, pad_id, max_len
  processed_data/val_dataset.pt    # optional
  processed_data/class_weights.pt  # 1D tensor of class weights for CE loss (id 0..K-1)

Outputs:
  runs/xlmrb_base_v1/              # best model, tokenizer, config, training logs
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
import numpy as np
import os
import warnings

# ---------------------------
# Dataset wrapper
# ---------------------------
class TensorDictDataset(Dataset):
    def __init__(self, pack: dict):
        # pack keys: input_ids [N,L], attention_mask [N,L], labels [N,L]
        self.input_ids = pack["input_ids"]
        self.attention_mask = pack["attention_mask"]
        self.labels = pack["labels"]
        # Optional
        self.rids = pack.get("rids", None)

        # Basic checks
        assert self.input_ids.shape == self.attention_mask.shape == self.labels.shape, \
            "Shapes of input_ids / attention_mask / labels must match"
        assert self.input_ids.dtype == torch.long and self.labels.dtype == torch.long, \
            "input_ids/labels must be torch.long"

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
        if self.rids is not None:
            item["rid"] = self.rids[idx]
        return item


# ---------------------------
# Weighted loss Trainer
# ---------------------------
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, ignore_index: int = -100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [B,L,C]
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device).to(dtype=logits.dtype),
            ignore_index=self.ignore_index
        )
        # flatten
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ---------------------------
# Metrics
# ---------------------------
def micro_metrics_from_logits(eval_preds, ignore_index=-100):
    """
    Compute micro precision/recall/F1/F0.2 ignoring positions with gold == -100.
    eval_preds is (logits, labels) where
      logits: np.array [N, L, C]
      labels: np.array [N, L]
    """
    logits, labels = eval_preds
    # predicted labels
    y_pred = logits.argmax(axis=-1).reshape(-1)
    y_true = labels.reshape(-1)

    mask = y_true != ignore_index
    if mask.sum() == 0:
        # Avoid division by zero if something went wrong
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "f0_2": 0.0,
            "num_eval_tokens": 0
        }
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    # micro metrics (exclude -100)
    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f0_2      = fbeta_score(y_true, y_pred, beta=0.2, average="micro", zero_division=0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f0_2": float(f0_2),
        "num_eval_tokens": int(mask.sum()),
    }


# ---------------------------
# Main
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pack", default="processed_data/train_dataset.pt")
    ap.add_argument("--val_pack",   default="processed_data/val_dataset.pt")
    ap.add_argument("--class_weights", default="processed_data/class_weights.pt")
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--output_dir", default="runs/xlmrb_base_v1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--eval_strategy", choices=["no","steps","epoch"], default="epoch")
    ap.add_argument("--save_strategy", choices=["steps","epoch"], default="epoch")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def infer_num_labels(*packs):
    # get max label id across provided packs, ignoring -100
    max_id = -1
    for pack in packs:
        if pack is None: 
            continue
        labels = pack["labels"].view(-1)
        mask = labels != -100
        if mask.any():
            max_id = max(max_id, int(labels[mask].max().item()))
    if max_id < 0:
        raise ValueError("Could not infer num_labels from dataset labels.")
    return max_id + 1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load packs
    train_pack = torch.load(args.train_pack)
    val_pack = torch.load(args.val_pack) if Path(args.val_pack).exists() else None

    # Datasets
    train_ds = TensorDictDataset(train_pack)
    eval_ds  = TensorDictDataset(val_pack) if val_pack is not None else None

    # Infer label space size from data (robust if no json mapping is provided)
    num_labels = infer_num_labels(train_pack, val_pack)

    # Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label={i: f"L{i}" for i in range(num_labels)},
        label2id={f"L{i}": i for i in range(num_labels)},
    )

    # Load class weights
    class_weights = torch.load(args.class_weights)
    if class_weights.ndim != 1 or class_weights.numel() != num_labels:
        warnings.warn(
            f"class_weights length ({class_weights.numel()}) != num_labels ({num_labels}). "
            "Will resize by truncating/padding with 1.0."
        )
        cw = torch.ones(num_labels, dtype=torch.float32)
        n = min(num_labels, class_weights.numel())
        cw[:n] = class_weights[:n].to(dtype=torch.float32)
        class_weights = cw

    # Trainer args
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.batch_size,
    #     num_train_epochs=args.epochs,
    #     learning_rate=args.learning_rate,
    #     weight_decay=args.weight_decay,
    #     warmup_ratio=args.warmup_ratio,
    #     logging_steps=args.logging_steps,
    #     evaluation_strategy=args.eval_strategy if eval_ds is not None else "no",
    #     save_strategy=args.save_strategy if eval_ds is not None else "epoch",
    #     load_best_model_at_end=eval_ds is not None,
    #     metric_for_best_model="f0_2",
    #     greater_is_better=True,
    #     # fp16=args.fp16,
    #     # bf16=args.bf16,
    #     fp16=False,
    #     bf16=False,
    #     seed=args.seed,
    #     report_to=[],  # disable W&B etc. for safety
    # )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy if eval_ds is not None else "no",
        save_strategy=args.save_strategy if eval_ds is not None else "epoch",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="f0_2",
        greater_is_better=True,

        # 强制 CPU
        no_cuda=True,
        dataloader_pin_memory=False,

        # 混合精度保持关掉
        fp16=False,
        bf16=False,

        seed=args.seed,
        report_to=[],  # 不用 W&B 等外部 logger
    )

    # Build trainer (with weighted loss)
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda p: micro_metrics_from_logits(p, ignore_index=-100),
    )

    # Train
    trainer.train()

    # Final eval (if available)
    if eval_ds is not None:
        metrics = trainer.evaluate()
        # Save metrics to file
        metrics_path = Path(args.output_dir) / "val_metrics.json"
        import json
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2, ensure_ascii=False)
        print(f"Validation metrics saved -> {metrics_path}")

    # Save final model
    trainer.save_model()  # saves to output_dir
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
