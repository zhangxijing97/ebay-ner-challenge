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
- **Produktart**: Precision = 0.70, Recall = 0.60  

We use \(F_\beta\) with \(\beta=0.2\) (precision-heavy):
\[
F_\beta=\frac{(1+\beta^2)\,P\,R}{\beta^2 P + R},\quad \beta=0.2 \Rightarrow \beta^2=0.04
\]
(For reference, \(F_1=\frac{2PR}{P+R}\).)

**Hersteller**
- \(F_{0.2}=\frac{1.04\times 0.90\times 0.80}{0.04\times 0.90+0.80}
=\frac{0.7488}{0.836}\approx \mathbf{0.896}\)
- \(F_{1}=\frac{2\times 0.90\times 0.80}{0.90+0.80}
=\frac{1.44}{1.70}\approx \mathbf{0.847}\)

**Produktart**
- \(F_{0.2}=\frac{1.04\times 0.70\times 0.60}{0.04\times 0.70+0.60}
=\frac{0.4368}{0.628}\approx \mathbf{0.696}\)
- \(F_{1}=\frac{2\times 0.70\times 0.60}{0.70+0.60}
=\frac{0.84}{1.30}\approx \mathbf{0.646}\)

If **Hersteller** appears **twice as often** as **Produktart**, the **category-level score** (weighted by aspect frequency) is:
- \(F_{0.2,\text{cat}}=\frac{0.896\times 2 + 0.696\times 1}{3}\approx \mathbf{0.829}\)
- \(F_{1,\text{cat}}=\frac{0.847\times 2 + 0.646\times 1}{3}\approx \mathbf{0.780}\)

Do the same for *Car Brake Component Kits*, then the **final leaderboard score** is the **average of the two category-level \(F_{0.2}\)** scores.

## Experiment Log

### Version 1
- **Approach**: Fine-tuned transformer (e.g., BERT) on 5k training set.  
- **Changes**: Added BIO tagging scheme for NER.  
- **Result**: TBD  