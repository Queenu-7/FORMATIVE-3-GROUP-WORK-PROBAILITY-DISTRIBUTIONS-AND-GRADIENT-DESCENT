# Formative 3: Probability Distributions, Bayesian Probability and Gradient Descent

Group project covering the EM algorithm, Bayes' Theorem on movie reviews, and gradient descent for linear regression.

---

## Team Members

| Name | Part(s) Led | Contribution |
|------|-------------|--------------|
| Uwera Ruth Queen | Part 1 and 3 | Completed all EM_CODE tasks and  manually calculated Gradient Descent up to iteration 2 |
| Natinael Borana | Part 3 and 4 | Completed all Gradient Descent code and plots, manually calculated iteration 2, and corrected errors in the overall manual calculations. |
| Maxime Kwizera | Part 2 | Completed calculations of Bayes’ code and report. |

Every member presents in **every** part. See the contribution PDF and Git logs for proof of work.

---

## Repository Structure

```
├── README.md
├── part1_em_algorithm/
│   ├── em_mothers.py
│   ├── GaltonFamilies.csv
│   └── (graphs: mean_split_vs_em.png, log_likelihood.png)
├── part2_bayesian/
│   ├── bayes_imdb_corrected.py
│   └── IMDB Dataset.csv
├── part3_manual_gradient_descent/
│   └── manual_calculations.pdf
├── part4_gradient_descent_code/
│   └── gradient_descent.py
├── notebook.ipynb
└── contributions.pdf
```

---

## How to Run

1. Clone the repo and open the folder.
2. Install the libraries: `pip install numpy matplotlib scipy`
3. Put each dataset in the same folder as its script.
4. Run a script, for example `python em_mothers.py`, or open `notebook.ipynb`.

**Google Colab:** upload the CSV from the file panel on the left first, otherwise you will get a `FileNotFoundError`.

---

## Part 1: Probability Distributions (EM Algorithm)

**Dataset:** Galton Families (heights in inches). **Choice:** Mothers vs Children.

We pool 205 mothers (one per family, so nobody is counted twice) and 934 children into one list of **1,139 heights** with **no labels**. EM then tries to find two bell curves (Gaussians) that explain the data.

### How EM works

| Step | What happens |
|------|--------------|
| **Initialization (iteration 0)** | Split at the global mean, then take the mean, variance and share of each pile as starting values. |
| **E-step (Expectation)** | For every height, use Bayes' Theorem to compute the probability it belongs to each group (the *responsibility*). |
| **M-step (Maximization)** | Recompute each group's mean, variance and weight, where every height counts in proportion to its responsibility. |
| **Stop** | When the log-likelihood barely improves (change below 1e-6). It never decreases. |

### Tracking table

| Iteration | μ1 | μ2 | σ1² | σ2² | π1 | π2 | Log-Likelihood |
|-----------|----|----|-----|-----|----|----|----------------|
| 0 (Initialization) | 63.5759 | 69.4718 | 3.4783 | 4.5477 | 0.5461 | 0.4539 | -3061.6755 |
| 1 | 63.7280 | 69.3527 | 4.1811 | 5.4578 | 0.5513 | 0.4487 | -3042.7613 |
| 2 | 63.8104 | 69.2956 | 4.5346 | 5.9142 | 0.5549 | 0.4451 | -3038.6870 |

Group 1 is the lower-height group and Group 2 is the higher-height group.

### Final result (converged after 331 iterations)

| | Mean | Variance | Weight |
|---|------|----------|--------|
| Group 1 (lower, mother-like) | 64.30 | 5.79 | 0.64 |
| Group 2 (higher, child-like) | 69.79 | 5.47 | 0.36 |

### Classifying a test height

Change `TEST_HEIGHT` (and `UNIT = "cm"` if needed) at the top of `em_mothers.py`. Example for **67 in**:

- P(Group 1, mother-like) = **0.6556**
- P(Group 2, child-like) = **0.3444**

### Why not just split at the global mean?

- **Overlap:** The groups overlap, but a hard line forces every height into one pile with 100% certainty. EM gives probabilities instead (soft assignments).
- **Biased estimates:** Cutting the bell curves chops off their tails. The split gave means of 63.58 / 69.47, while EM found 64.30 / 69.79.
- **Wrong group sizes:** The split forces roughly 55/45. EM found the real proportions are about 64/36.
- **No uncertainty:** A line cannot say "65% likely". EM can, and its log-likelihood shows how well the model fits.

We still use the mean split as iteration 0, as a starting point that EM then improves.

### Key finding (important caveat)

After the fact, we revealed the hidden labels:

| True group | In Group 1 | In Group 2 |
|------------|-----------:|-----------:|
| Mother | 195 | 10 |
| Daughter | 421 | 32 |
| Son | 124 | 357 |

EM mostly separated **women (mothers and daughters) from men (sons)**, not mothers from children. Half the children are sons, who are as tall as fathers. So "mother-like" and "child-like" are interpretations, not true identities.

---

## Part 2: Bayesian Probability (IMDb Reviews)

**Dataset:** IMDb 50K Movie Reviews (25,000 positive, 25,000 negative).
**We compute:** **P(Positive | keyword)** only.

### Bayes' Theorem

```
P(Positive | keyword) = P(keyword | Positive) × P(Positive) / P(keyword)
```

### Keywords and justification

- **Positive:** excellent, amazing, wonderful, brilliant (words reviewers use when praising a film).
- **Negative:** terrible, boring, awful, worst (words reviewers use when criticizing one).

### Results

Prior P(Positive) = **0.5000** for every keyword.

| Keyword | Likelihood P(keyword \| Positive) | Marginal P(keyword) | Posterior P(Positive \| keyword) |
|---------|-----------------------------------|---------------------|----------------------------------|
| excellent | 0.1147 | 0.0710 | **0.8074** |
| amazing | 0.0672 | 0.0432 | **0.7780** |
| wonderful | 0.0903 | 0.0556 | **0.8122** |
| brilliant | 0.0635 | 0.0418 | **0.7601** |
| terrible | 0.0153 | 0.0540 | **0.1418** |
| boring | 0.0237 | 0.0610 | **0.1940** |
| awful | 0.0114 | 0.0576 | **0.0985** |
| worst | 0.0164 | 0.0887 | **0.0927** |

### Method notes

- Text is lowercased, and punctuation and `<br />` tags are removed, so "terrible." still matches "terrible".
- Each review counts **once** per keyword, however many times it appears.
- A direct count (positive reviews with keyword / all reviews with keyword) matches the Bayes result, as a sanity check.

### Takeaways

- Positive keywords give posteriors of about 76-81%, negative keywords about 9-19%.
- "worst" and "awful" are the strongest negative signals. "boring" is the weakest, probably because of phrases like "never boring". Our code does not handle negation.

---

## Part 3: Gradient Descent (Manual Calculation)

**Model:** y = m₁x₁ + m₂x₂ + b, with initial m = [-1, 2], b = [1, 1].
**Data:** x = ((1, 3), (4, 10)), y = (5, 6).

All steps use **matrix multiplication**, not scalar arithmetic. The number of updates equals the number of group members, and each member completed at least one (see the contribution PDF).

### Setup in matrix form

```
X = [[1,  3],        m = [[-1],        y = [[5],
     [4, 10]]             [ 2]]             [6]]
```

**Predictions:** ŷ = Xm + b

| | Calculation | Result |
|---|---|---|
| Sample 1 | (1)(-1) + (3)(2) + 1 | 6 |
| Sample 2 | (4)(-1) + (10)(2) + 1 | 17 |

So ŷ = [6, 17], the errors (ŷ - y) are [1, 11], and the starting MSE is J = (1² + 11²) / 2 = **61**.

### Gradient derivation (chain rule)

Cost: J(m, b) = (1/N) Σ (ŷᵢ - yᵢ)², where ŷ = Xm + b.

Let e = ŷ - y. By the chain rule, ∂J/∂m = (∂J/∂e)(∂e/∂m):

```
∂J/∂e = (2/N) e
∂e/∂m = X        →   ∂J/∂m = (2/N) Xᵀ (ŷ - y)
∂e/∂b = 1        →   ∂J/∂b = (2/N) Σ (ŷ - y)
```

### Update rule

```
m ← m - α · ∂J/∂m
b ← b - α · ∂J/∂b
```

### Gradients at the start

With the 1/N convention above:

```
∂J/∂m = [45, 113]      ∂J/∂b = 12
```

### Updates (learning rate α = 0.01)

| Update | m₁ | m₂ | b | MSE (J) |
|:------:|----:|----:|----:|--------:|
| 0 (start) | -1.0000 | 2.0000 | 1.0000 | 61.0000 |
| 1 | -1.4500 | 0.8700 | 0.8800 | 6.8450 |
| 2 | -1.3316 | 1.1808 | 0.9318 | 2.7238 |
| 3 | -1.3696 | 1.0952 | 0.9362 | 2.3904 |

Example (update 1): m₁ = -1 - 0.01(45) = **-1.45**, m₂ = 2 - 0.01(113) = **0.87**, b = 1 - 0.01(12) = **0.88**.

Every update is worked out step by step in `manual_calculations.pdf`: predictions, error, both gradients, and the new m and b, with the member who completed it named.

### Trend

- The error J drops from 61 to 6.85 after the first update and keeps falling, so m and b are moving in a direction that **reduces the error**.
- m₁ becomes more negative, m₂ drops from 2 to about 1.1, and b dips slightly at first (0.88) and then rises.
- The first update is a big jump because the starting gradient is large. After that the steps get smaller.

---

## Part 4: Gradient Descent (Code)

The manual calculations from Part 3 are converted to Python, with every step visible (predict, cost, gradient, update) and no heavy abstraction. Libraries: **numpy, scipy, matplotlib**.

### What the code does

| Function / step | Purpose |
|-----------------|---------|
| `calculate_prediction(X, theta)` | ŷ = Xm + b using matrix multiplication (`theta = [m₁, m₂, b]`) |
| `mse_cost_function(theta, X, y)` | J = (1/N) eᵀe, where e = ŷ - y |
| `calculate_derivative(cost, theta, X, y)` | Uses SciPy's `approx_fprime` to compute the gradient of any cost function |
| Update loop | `theta = theta - alpha * grads`, repeated for 30 epochs, saving m, b and cost each time |

### Settings

- Learning rate α = **0.01**, **30** epochs.
- Start: m = [-1, 2], b = 1 (the same b is added to every prediction).

### Check against the manual work

| | Value |
|---|---|
| Initial predictions | [6, 17] |
| Initial MSE | 61.0000 |
| Initial gradient (SciPy) | [45, 113, 12] |

These match the hand calculations. (SciPy's numerical derivative shows tiny rounding noise such as 113.00000095.)

### Results after 30 updates

| Parameter | Value |
|-----------|------:|
| m₁ | -1.5717 |
| m₂ | 1.1095 |
| b | 1.6654 |
| Final MSE | 1.356827 |

| | Prediction (ŷ) | Actual (y) |
|---|---:|---:|
| Sample 1 | 3.42232721 | 5 |
| Sample 2 | 6.47392283 | 6 |

The cost fell from **61 to 1.82** and was still decreasing slowly at epoch 30, so the model had not fully converged. More epochs or a larger α would reduce it further.

### Plots

1. **Parameters vs iterations:** m₁, m₂ and b across all epochs.
2. **MSE vs iterations:** a steep drop in the first few epochs, then a slow, steady decline.

---

## Notes for the Presentation

- **Time:** 15 minutes. Every member speaks in every part.
- **Part 1:** show the tracking table, answer the "mean split" question, classify the coach's test height, then mention the women-vs-men finding.
- **Part 2:** show the 8 keyword tables and explain why the posteriors are high even though the likelihoods are low (the marginal is small too).
- **Part 3:** show the chain rule derivation and each member's update.
- **Part 4:** run the code and explain the two plots.

## Libraries Used

| Part | Libraries |
|------|-----------|
| 1 | numpy, matplotlib, csv (built-in) |
| 2 | numpy, matplotlib, csv (built-in) |
| 3 | handwritten (PDF) |
| 4 | numpy, scipy, matplotlib |

No machine-learning libraries (scikit-learn, etc.) are used anywhere.

