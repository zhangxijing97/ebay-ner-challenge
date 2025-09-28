# eBay NER Challenge

This repository contains my work for the **eBay Named Entity Recognition (NER) Challenge** on [EvalAI](https://eval.ai/web/challenges/challenge-page/2508/overview).

The goal of the competition is to extract structured product aspects from German eBay item titles in the domains of:
- **Car Brake Component Kits**  
- **Car Engine Timing Kits**

## How the Final Score is Calculated
1. For each **aspect name** inside a category:
   - Compute **Precision** = correct predictions / all predictions  
   - Compute **Recall** = correct predictions / all true labels  
   - Compute **F<sub>0.2</sub>** = weighted combination giving more weight to precision.
2. Weight each aspect by its frequency in the dataset (common aspects have more impact).  
3. Sum scores across all aspects to get a **category-level score**.  
4. The **final score** = mean of the two category-level scores.

### Example — How the score is computed

Suppose in *Car Engine Timing Kits* we evaluate two aspects:

**Hersteller**: Precision = 0.90, Recall = 0.80  
- F0.2 = (1 + β^2）* P * R) / (β^2 * P + R) = (1.04 * 0.90 * 0.80) / (0.04 * 0.90 + 0.80) = 0.7488 / 0.836 = **0.90**  
- F1 = (2 * 0.90 * 0.80) / (0.90 + 0.80) = 1.44 / 1.70 = **0.85**

**Produktart**: Precision = 0.70, Recall = 0.60  
- F0.2 = (1 + β^2）* P * R) / (β^2 * P + R) = (1.04 * 0.70 * 0.60) / (0.04 * 0.70 + 0.60) = 0.4368 / 0.628 = **0.70**  
- F1 = (2 * 0.70 * 0.60) / (0.70 + 0.60) = 0.84 / 1.30 = **0.65**

If **Hersteller** appears **twice as often** as **Produktart**, the **category-level score** (weighted by aspect frequency) is:

- F0.2 = (0.90 * 2 + 0.70 * 1) / 3 = (1.80 + 0.70) / 3 = **0.83**  
- F1 = (0.85 * 2 + 0.65 * 1) / 3 = (1.70 + 0.65) / 3 = **0.78**

Do the same for *Car Brake Component Kits*, then the **final leaderboard score** is the **average of the two category-level F0.2 scores**.

## Experiment Log

### How our F1 / F0.2 are computed

Colab: <https://colab.research.google.com/drive/1-QZhYnF119t9eI9yRigq4q2MY8T8hfOP?usp=drive_link>  

**Implementation detail**: The full evaluation logic is implemented inside  
```python
def evaluate_logits(model, data_loader, device, tag2idx, beta_for_final=0.2):
```

> **Note**: The evaluation method described below is **not identical to the official competition scoring**.  
> As a result, the absolute F1/F0.2 values we obtain may differ from leaderboard scores.  
> However, this method provides a **consistent and reliable way to compare models** during experiments, so that we can roughly judge which setups perform better or worse.

- We turn tags into **entities** per sequence: keep the first tag of a run (non-`O`), and treat later repeats as “continuation” (`""`). Then we join its tokens into one entity string.
- We count entities **per (record, category, aspect)** using a Counter for **gold** and **pred**.
- For each (record, category, aspect):
  - **TP** = overlap of gold vs pred counts (per value, use min)
  - **FP** = predicted counts − overlap
  - **FN** = gold counts − overlap
- **Precision** = TP/(TP+FP), **Recall** = TP/(TP+FN)
- **F1** (β=1): `(1+1^2)*P*R/(1^2*P+R) = 2PR/(P+R)`
- **F0.2** (β=0.2): `(1+β²)PR/(β²P+R)` → puts **more weight on Precision** than Recall
- **Category score** = weighted average of aspect F-scores (weights = gold entity counts for that aspect).
- **Final score** = average of the two category scores.

### EXP-001 — Baseline BiLSTM
Colab: <https://colab.research.google.com/drive/1-QZhYnF119t9eI9yRigq4q2MY8T8hfOP?usp=drive_link>  
Date: 2025-09-26 (PT)  
Config: `BiLSTMWithCategory(emb=128, cat=10, hidden=256, layers=1, no dropout)`  
Train: `Adam(lr=1e-3)`, `CE(ignore PAD)`, epochs=50, bs=32, no clipping/scheduler/seed  
Split: train/val = 90%/10%

Best: **Epoch 37**  
- final_F1 **0.8725**, final_F0.2 **0.8773**  
- c1_F1 **0.9007**, c2_F1 **0.8444**

### EXP-002 — BiLSTM GradClip Sweep (one experiment)

Gradient clipping: Limiting the gradient’s magnitude (e.g., its norm) to a preset threshold during backpropagation to prevent exploding gradients and stabilize training.  
Colab: <https://colab.research.google.com/drive/12Y2OvUpeR2HB3PuwF6qw8c03itXj3Kaj?usp=drive_link>  
Date: 2025-09-26 (PT)  
Config: `BiLSTMWithCategory(emb=128, cat=10, hidden=256, layers=1, dropout=0.0)`  
Train: `Adam(lr=1e-3)`, `CrossEntropy(ignore PAD)`, **epochs=50**, **bs=32**, **fixed seed**, no scheduler  
Split: train/val = 90%/10%  
Variants: `grad_clip_norm ∈ {0.25, 0.5, 1.0, 2.0, 5.0}`

**Best-per-variant (by final_F0.2)**

| grad_clip_norm | best_epoch | final_F1 | final_F0.2 | c1_F1 | c2_F1 | checkpoint |
|---:|---:|---:|---:|---:|---:|:--|
| 0.25 | 33 | 0.8741 | 0.8787 | 0.8936 | 0.8546 | `bilstm_gc0.25.pt` |
| 0.5  | 21 | 0.8758 | 0.8835 | 0.9049 | 0.8466 | `bilstm_gc0.5.pt`  |
| 1.0  | 19 | 0.8727 | 0.8774 | 0.9015 | 0.8439 | `bilstm_gc1.0.pt`  |
| 2.0  | 37 | 0.8807 | 0.8874 | 0.9004 | 0.8610 | `bilstm_gc2.0.pt` |
| 5.0  | 9  | 0.8775 | 0.8853 | 0.9042 | 0.8508 | `bilstm_gc5.0.pt`  |

**Overall winner:** `grad_clip_norm=2.0` → final_F0.2 **0.8874** (best overall)

**Repeat run #2**

| grad_clip_norm | best_epoch | final_F1 | final_F0.2 | c1_F1 | c2_F1 | checkpoint |
|---:|---:|---:|---:|---:|---:|:--|
| 0.25 | 46 | 0.8799 | 0.8825 | 0.9048 | 0.8551 | `bilstm_gc0.25.pt` |
| 0.5  | 40 | 0.8710 | 0.8774 | 0.8944 | 0.8475 | `bilstm_gc0.5.pt`  |
| 1.0  | 50 | 0.8755 | 0.8770 | 0.9048 | 0.8461 | `bilstm_gc1.0.pt`  |
| 2.0  | 41 | 0.8699 | 0.8753 | 0.9075 | 0.8323 | `bilstm_gc2.0.pt` |
| 5.0  | 46 | 0.8767 | 0.8811 | 0.9000 | 0.8534 | `bilstm_gc5.0.pt`  |

**Overall winner:** `grad_clip_norm=0.25` → final_F0.2 **0.8825** (best overall)

**Repeat run #3**

| grad_clip_norm | best_epoch | final_F1 | final_F0.2 | c1_F1 | c2_F1 | checkpoint |
|---:|---:|---:|---:|---:|---:|:--|
| 0.25 | 10 | 0.8812 | 0.8906 | 0.9046 | 0.8577 | `bilstm_gc0.25.pt` |
| 0.5  | 36 | 0.8806 | 0.8862 | 0.9000 | 0.8611 | `bilstm_gc0.5.pt`  |
| 1.0  | 46 | 0.8771 | 0.8820 | 0.9021 | 0.8521 | `bilstm_gc1.0.pt`  |
| 2.0  | 11 | 0.8768 | 0.8850 | 0.9005 | 0.8532 | `bilstm_gc2.0.pt` |
| 5.0  | 17 | 0.8803 | 0.8876 | 0.9078 | 0.8528 | `bilstm_gc5.0.pt`  |

**Overall winner:** `grad_clip_norm=0.25` → final_F0.2 **0.8906** (best overall)

#### Notes
- Using **final_F0.2** favors settings that maintain higher recall at stricter precision weighting; under this lens, **0.25** and **5.0** often shine early, while **2.0** remains a solid all-rounder.
- Late-epoch variance still appears at larger clips; early stopping by **final_F0.2** could be beneficial.

#### Next micro-step (keep everything else the same)
- Start from **grad_clip_norm=2.0**; add **dropout=0.3** after the LSTM output (before the FC).  
- Run `layers=1, dropout=0.3`. If helpful, try `layers=2, dropout=0.3`.  
- Goal: lift **c2_F1** and smooth late-epoch fluctuations.

### EXP-003 — BiLSTM + GradClip + Dropout

Lightweight regularization: Small, low-cost techniques (e.g., dropout, weight decay, early stopping) that gently constrain the model to reduce overfitting and improve generalization without architectural changes.  
Colab: <https://colab.research.google.com/drive/1FQwhsm0mMxjFnu36A0pgrRllFaBRainb?usp=drive_link>  
Date: 2025-09-26 (PT)  
Config: `BiLSTMWithCategory(emb=128, cat=10, hidden=256, layers=1)`  
Train: Adam, CE, epochs=50, bs=32, gradient clipping  

| Run  | Dropout | Clip | Best Ep | final_F1 | final_F0.2 | best_F1 | best_F0.2 |
|:---:|:------:|:---:|:------:|:-------:|:---------:|:------:|:---------:|
| 09 | 0.4 | 0.25 | **28** | 0.8934 | 0.8989 | 0.8972 | **0.9055** |
| 11 | 0.4 | 1.0  | **26** | 0.8953 | 0.9005 | 0.8956 | **0.9033** |
| 13 | 0.5 | 0.25 | **41** | 0.8902 | 0.8967 | 0.8973 | **0.9041** |
| 14 | 0.5 | 0.5  | **49** | 0.8955 | 0.9023 | 0.8981 | **0.9039** |
| 15 | 0.5 | 1.0  | **35** | 0.8946 | 0.8999 | 0.8988 | **0.9046** |

- Smaller clip (0.25) + 0.4 dropout → faster convergence, stronger precision-weighted F0.2.
- 0.5 dropout + clip 1.0 → later peak, best balanced F1.
- clip 2.0 is generally weaker.
- Category_2 is more sensitive and benefits from the later-peaking setup.

### EXP-004 — BiLSTM + GradClip + Dropout + AdamW + Weight Decay
Colab: <https://colab.research.google.com/drive/1ZGdPHmAt9doK_4WWM4SC9DicjbJ-QJkj?usp=drive_link>  
Date: 2025-09-27 (PT)  
Config: `BiLSTMWithCategory(emb=128, cat=10, hidden=256, layers=2, dropout∈{0.2–0.4})`  
Train: `AdamW(lr∈{5e-4–1e-3})`, `CE(ignore PAD)`, `weight_decay∈{0.003–0.01}`, `grad_clip∈{0.25–1.0}`, `ReduceLROnPlateau`, epochs≈40–50, bs=32  
Split: train/val = 90%/10%

Best (across trials): **[COARSE RUN 14]**  
- Config: `lr=0.001 | dropout=0.4 | clip=0.25 | weight_decay=0.005 | epochs=40`  
- [Eval] c1_F0.2 **0.9278**, c1_F1 **0.9212**, c2_F0.2 **0.8949**, c2_F1 **0.8805**  
- **final_F0.2 0.9114**, **final_F1 0.9008**

Summary: Across multiple settings, `AdamW + weight decay + scheduler` **did not consistently beat** the baseline F0.2 (baseline best ≈ **0.9055**). Only the above single run performed notably well.  
Decision: **Not adopted as mainline for now.** Continue with “BiLSTM + GradClip + Dropout”; revisit this route after testing (i) **no-decay for embeddings/bias** and (ii) **LR scheduling driven by F0.2**.