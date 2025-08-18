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

- **Hersteller**: Precision = 0.90, Recall = 0.80  
  - F0.2 = (1.04 * 0.90 * 0.80) / (0.04 * 0.90 + 0.80) = 0.7488 / 0.836 = **0.90**  
  - F1 = (2 * 0.90 * 0.80) / (0.90 + 0.80) = 1.44 / 1.70 = **0.85**

- **Produktart**: Precision = 0.70, Recall = 0.60  
  - F0.2 = (1.04 * 0.70 * 0.60) / (0.04 * 0.70 + 0.60) = 0.4368 / 0.628 = **0.70**  
  - F1 = (2 * 0.70 * 0.60) / (0.70 + 0.60) = 0.84 / 1.30 = **0.65**

If **Hersteller** appears **twice as often** as **Produktart**, the **category-level score** (weighted by aspect frequency) is:

- F0.2 = (0.90 * 2 + 0.70 * 1) / 3 = (1.80 + 0.70) / 3 = **0.83**  
- F1 = (0.85 * 2 + 0.65 * 1) / 3 = (1.70 + 0.65) / 3 = **0.78**

Do the same for *Car Brake Component Kits*, then the **final leaderboard score** is the **average of the two category-level F0.2 scores**.

## Experiment Log

### Version 1
- **Approach**: Fine-tuned transformer (e.g., BERT) on 5k training set.  
- **Changes**: Added BIO tagging scheme for NER.  
- **Result**: TBD  