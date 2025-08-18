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

### Example
Suppose in *Car Engine Timing Kits*:
- Aspect **Hersteller**: Precision = 0.9, Recall = 0.8 → F<sub>0.2</sub> ≈ 0.88  
- Aspect **Produktart**: Precision = 0.7, Recall = 0.6 → F<sub>0.2</sub> ≈ 0.68  

If *Hersteller* appears twice as often as *Produktart*, then category score ≈  
`(0.88*2 + 0.68*1) / 3 = 0.81`.

Do the same for *Car Brake Component Kits*, then take the **average of the two categories** to get the final leaderboard score.

## Experiment Log

### Version 1
- **Approach**: Fine-tuned transformer (e.g., BERT) on 5k training set.  
- **Changes**: Added BIO tagging scheme for NER.  
- **Result**: TBD  