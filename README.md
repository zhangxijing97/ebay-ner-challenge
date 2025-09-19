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

### Deep Neural Network (DNN)

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

### Recurrent Neural Network (RNN)

#### 1. Forward Pass  
At each time step `t`, the RNN takes the current input `x_t` and the previous hidden state `h_{t-1}`:  

Hidden state update:  
`a_t = W_x * x_t + W_h * h_{t-1} + b`  

- `x_t` → the input at step t (e.g., the current word or number)  
- `W_x` → weight matrix that scales the current input  
- `h_{t-1}` → hidden state from the previous time step (the "memory")  
- `W_h` → weight matrix that controls how much past memory influences the current step  
- `b` → bias term, shifts the result  

Activation:  
`h_t = tanh(a_t)`  

- `tanh()` → squashes values into (-1, 1), introduces nonlinearity, and keeps hidden state stable  
- `h_t` → updated hidden state (combines current input and past context)  

Output at time t:  
`y_t = W_y * h_t + c`  

- `W_y` → weight mapping hidden state to output  
- `c` → output bias  
- `y_t` → model’s prediction at step t  

👉 Key difference from DNN:  
- DNN only looks at input `x`  
- RNN uses both `x_t` and the memory `h_{t-1}`, so it can "remember" context from earlier steps.  

#### 2. Loss Function (Sequence MSE example)  
For a sequence of length T, with target outputs `y*_t`:  

`Loss = Σ ( 0.5 * (y_t - y*_t)^2 )` for `t = 1...T`

#### 3. Backward Pass (Backpropagation Through Time, BPTT)  
- Compute gradient at each step t:  
  `δa_t = (∂Loss/∂y_t * W_y + δa_{t+1} * W_h) * (1 - h_t^2)`  

- Use these to calculate parameter gradients:  
  - `∇W_x = Σ δa_t * x_t`  
  - `∇W_h = Σ δa_t * h_{t-1}`  
  - `∇W_y = Σ h_t * (y_t - y*_t)`  
  - `∇b  = Σ δa_t`  
  - `∇c  = Σ (y_t - y*_t)`  

#### 4. Gradient Descent Updates  
Each parameter θ (weights and biases) is updated as:  

`θ = θ - α * ∂Loss / ∂θ`  

Examples:  
- `W_x = W_x - α * ∇W_x`  
- `W_h = W_h - α * ∇W_h`  
- `W_y = W_y - α * ∇W_y`  
- `b   = b   - α * ∇b`  
- `c   = c   - α * ∇c`  

#### ✅ Example (Sequence [1,2,3] → Predict [2,3,4])  

- **Initialization**
  - Inputs: `x = [1, 2, 3]`
  - Initial hidden state: `h0 = 0`
  - Parameters: `W_x = 0.6`, `W_h = 0.5`, `W_y = 1.0`, `b = 0`, `c = 0`
  - Update/Output rules:
    - `a_t = W_x * x_t + W_h * h_{t-1} + b`
    - `h_t = tanh(a_t)`
    - `y_t = W_y * h_t + c`

- **Step-by-step (initial forward pass)**  

  - t=1:  
    `a1 = W_x * x1 + W_h * h0 + b = 0.6*1 + 0.5*0 + 0 = 0.6`  
    `h1 = tanh(a1) = tanh(0.6) ≈ 0.537`  
    `y1 = W_y * h1 + c = 1.0*0.537 + 0 = 0.537 ≈ 0.54`  

  - t=2:  
    `a2 = W_x * x2 + W_h * h1 + b = 0.6*2 + 0.5*0.537 + 0 = 1.4685`  
    `h2 = tanh(a2) = tanh(1.4685) ≈ 0.899`  
    `y2 = W_y * h2 + c = 1.0*0.899 + 0 = 0.899 ≈ 0.90`  

  - t=3:  
    `a3 = W_x * x3 + W_h * h2 + b = 0.6*3 + 0.5*0.899 + 0 = 2.2497`  
    `h3 = tanh(a3) = tanh(2.2497) ≈ 0.978`  
    `y3 = W_y * h3 + c = 1.0*0.978 + 0 = 0.978 ≈ 0.98`  

- Initial predictions: `[0.54, 0.90, 0.98]` (far from targets)  
- After 1 update: `[1.86, 2.19, 2.22]` (loss dropped from **7.84 → 1.93**)  
- After 10 updates: `[2.12, 3.13, 3.44]` (loss ≈ **0.17**)  

| Epoch | Predictions (y1, y2, y3)      | Loss   |
|-------|-------------------------------|--------|
| 0     | [0.54, 0.90, 0.98]            | 7.84   |
| 1     | [1.86, 2.19, 2.22]            | 1.93   |
| 2     | [1.95, 2.63, 2.71]            | 0.95   |
| 3     | [2.05, 2.85, 3.00]            | 0.52   |
| 4     | [2.08, 2.98, 3.18]            | 0.36   |
| 5     | [2.12, 3.06, 3.28]            | 0.27   |
| 6     | [2.16, 3.11, 3.35]            | 0.22   |
| 7     | [2.19, 3.14, 3.39]            | 0.19   |
| 8     | [2.23, 3.12, 3.32]            | 0.26   |
| 9     | [2.16, 3.12, 3.38]            | 0.21   |
| 10    | [2.12, 3.13, 3.44]            | 0.17   |

👉 RNN gradually learned the sequence rule (`next number = +1`) by **remembering past inputs via hidden states**.

####  RNN Parameters Cheat Sheet

| Parameter | Controls | ↑ Increase | ↓ Decrease |
|-----------|----------|------------|------------|
| **W_x** (input weights) | Impact of current input `x_t` | Focus more on current input | Rely more on past state |
| **W_h** (recurrent weights) | Impact of past state `h_{t-1}` | Stronger memory, long-term context | Weaker memory, short-term focus (like DNN) |
| **W_y** (output weights) | Map hidden state `h_t` → output `y_t` | Stronger, sharper outputs | Softer, less confident outputs |
| **b** (hidden bias) | Hidden state baseline | Easier to activate | Harder to activate |
| **c** (output bias) | Output baseline | Outputs shift higher | Outputs shift lower |

### Long Short-Term Memory (LSTM)

#### 1. Forward Pass  
At each time step `t`, the LSTM takes the current input `x_t`, the previous short-term memory (hidden state) `h_{t-1}`, and the previous long-term memory (cell state) `C_{t-1}`.

![LSTM Structure](image/lstm.png)

**(a) Forget Gate – Blue (% Long-Term To Remember)**  
`f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`  

- Decides how much of the old long-term memory to keep  
- `σ` → sigmoid function, outputs between (0,1)  
- If `f_t ≈ 1` → keep most of `C_{t-1}`  
- If `f_t ≈ 0` → forget most of `C_{t-1}`  

**(b) Input Gate + Candidate Memory – Green + Yellow**  
`i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`  
`\tilde{C}_t = tanh(W_C · [h_{t-1}, x_t] + b_C)`  

- Input Gate (green) controls **how much new info enters**  
- Candidate Memory (yellow, via tanh) generates potential new content  

**(c) Cell State Update – Combination**  
`C_t = f_t * C_{t-1} + i_t * \tilde{C}_t`  

- Combines “forgotten old” and “added new” information  
- Result is the **new long-term memory**  

**(d) Output Gate – Purple (% Potential Memory To Remember)**  
`o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`  
`h_t = o_t * tanh(C_t)`  

- Decides what part of the updated long-term memory is exposed as short-term memory  
- Final outputs of this step:  
  - `C_t` → new long-term memory  
  - `h_t` → new short-term memory / hidden state  

---

#### 2. Loss Function (Sequence Example with MSE)  
For a sequence of length T, with targets `y*_t`:  

`Loss = Σ (0.5 * (y_t - y*_t)^2)` for `t = 1...T`

---

#### 3. Backward Pass (Backpropagation Through Time with Gates)  
- Compute gradients through each gate:  
  - `δf_t, δi_t, δo_t, δ\tilde{C}_t` from chain rule  
- Update parameters:  
  - `∇W_f, ∇W_i, ∇W_o, ∇W_C`  
  - `∇b_f, ∇b_i, ∇b_o, ∇b_C`  

---

#### 4. Gradient Descent Updates  
Each parameter θ is updated as:  

`θ = θ - α * ∂Loss / ∂θ`  

Examples:  
- `W_f = W_f - α * ∇W_f`  
- `W_i = W_i - α * ∇W_i`  
- `W_C = W_C - α * ∇W_C`  
- `W_o = W_o - α * ∇W_o`  
- Biases updated similarly: `b_f, b_i, b_C, b_o`  

---

#### ✅ Example (Tiny Walkthrough)  
Suppose:  
- Input at step t: `x_t = 1`  
- Short-term memory: `h_{t-1} = 1`  
- Long-term memory: `C_{t-1} = 2`  
- Forget Gate formula (from diagram): `f_t = σ(2.70*h + 1.63*x + 1.62)`  

**Step**  
- Forget Gate: `f_t ≈ 0.997` → keep ~99.7% of `C_{t-1}`  
- Input Gate + Candidate: add controlled new info via `i_t * \tilde{C}_t`  
- Cell State: `C_t = f_t * 2 + i_t * \tilde{C}_t` ≈ 1.99 + extra new info  
- Output Gate: `h_t = o_t * tanh(C_t)` → produces new short-term memory  

👉 Result: LSTM preserves old memory (because Forget Gate is high) but also integrates new input.  

---

#### LSTM Parameters Cheat Sheet  

| Parameter | Controls | ↑ Increase | ↓ Decrease |
|-----------|----------|------------|------------|
| **W_f** (forget weights, blue) | How much old memory to keep | Less forgetting | More forgetting |
| **W_i** (input weights, green) | How much new info to add | More update from input | Less new info written |
| **W_C** (candidate weights, yellow) | What the new memory looks like | Stronger candidate content | Weaker candidate content |
| **W_o** (output weights, purple) | How much memory goes to output | Stronger hidden signal | Weaker hidden signal |
| **b_f, b_i, b_C, b_o** | Bias shifts each gate | Easier/harder to activate gates | — |