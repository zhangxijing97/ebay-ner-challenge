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

### Version 1
- **Approach**: Fine-tuned transformer (e.g., BERT) on 5k training set.  
- **Changes**: Added BIO tagging scheme for NER.  
- **Result**: TBD  

### DNN

#### 1. Forward Pass  
Input layer:  
`z1 = w11 * x1 + w21 * x2 + b1`  
`h1 = ReLU(z1)`  

`z2 = w12 * x1 + w22 * x2 + b2`  
`h2 = ReLU(z2)`  

Output layer (sigmoid activation):  
`z_out = v1 * h1 + v2 * h2 + b_out`  
`ŷ = 1 / (1 + exp(-z_out))`  

#### 2. Loss Function (Binary Cross-Entropy)  
`Loss = - ( y * log(ŷ) + (1 - y) * log(1 - ŷ) )`  

#### 3. Backward Pass + Gradient Descent  
Each parameter θ (weights and biases) is updated as:  

`θ = θ - α * ∂Loss / ∂θ`  

Examples:  
- `w11 = w11 - α * ∂Loss / ∂w11`  
- `w21 = w21 - α * ∂Loss / ∂w21`  
- `v1 = v1 - α * ∂Loss / ∂v1`  
- `b_out = b_out - α * ∂Loss / ∂b_out`  

####  Summary
1. Forward Pass → compute prediction `ŷ`  
2. Loss → measure difference between prediction and true label `y`  
3. Backward Pass → compute gradients  
4. Gradient Descent → update all weights and biases to reduce loss  

### BiLSTM

### BiLSTM