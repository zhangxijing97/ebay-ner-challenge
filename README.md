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
Colab: <https://colab.research.google.com/drive/12Y2OvUpeR2HB3PuwF6qw8c03itXj3Kaj?usp=drive_link>
Date: 2025-09-26 (PT)  
Config: `BiLSTMWithCategory(emb=128, cat=10, hidden=256, layers=1, dropout=0.0)`  
Train: `Adam(lr=1e-3)`, `CrossEntropy(ignore PAD)`, **epochs=50**, **bs=32**, **fixed seed**, no scheduler  
Split: train/val = 90%/10%  
Variants: `grad_clip_norm ∈ {0.25, 0.5, 1.0, 2.0, 5.0}`

**Best-per-variant (by final_F1)**

| grad_clip_norm | best_epoch | final_F1 | final_F0.2 | c1_F1 | c2_F1 | checkpoint |
|---:|---:|---:|---:|---:|---:|:--|
| 0.25 | 33 | **0.8741** | 0.8787 | 0.8936 | 0.8546 | `bilstm_gc0.25.pt` |
| 0.5  | 21 | **0.8758** | 0.8835 | 0.9049 | 0.8466 | `bilstm_gc0.5.pt`  |
| 1.0  | 19 | **0.8727** | 0.8774 | 0.9015 | 0.8439 | `bilstm_gc1.0.pt`  |
| 2.0  | 37 | **0.8807** | 0.8874 | 0.9004 | **0.8610** | `bilstm_gc2.0.pt` |
| 5.0  | 9  | **0.8775** | 0.8853 | 0.9042 | 0.8508 | `bilstm_gc5.0.pt`  |

**Overall winner:** `grad_clip_norm=2.0` → final_F1 **0.8807** (best overall) with the strongest category-2 score (c2_F1 **0.8610**).

#### Notes
- Very small clipping (0.25) stabilizes early recall (F0.2) but caps overall F1 later.
- Very large clipping (5.0) can boost early recall but shows more late-epoch variance.
- Mild overfitting emerges after ~epoch 25–30 across several settings (loss ↓ while F1 fluctuates).

#### Next micro-step (keep everything else the same)
- Start from **grad_clip_norm=2.0**; add **dropout=0.3** after the LSTM output (before the FC).  
- Run `layers=1, dropout=0.3`. If helpful, try `layers=2, dropout=0.3`.  
- Goal: lift **c2_F1** and smooth late-epoch fluctuations.

Artifacts: `bilstm_gc0.25.pt`, `bilstm_gc0.5.pt`, `bilstm_gc1.0.pt`, `bilstm_gc2.0.pt`, `bilstm_gc5.0.pt`