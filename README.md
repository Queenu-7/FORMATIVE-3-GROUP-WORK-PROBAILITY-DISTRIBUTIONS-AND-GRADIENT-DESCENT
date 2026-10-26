# Formative 3: Probability Distributions, Bayesian Probability and Gradient Descent

Group project covering the EM algorithm, Bayes' Theorem on movie reviews, and gradient descent for linear regression.

---

## Team Members

| Name | Part(s) Led | Contribution |
|------|-------------|--------------|
| Uwera Ruth Queen | Part 1, 2, and 3 | Completed all EM_CODE tasks, manually calculated Gradient Descent up to iteration 2 and implemented Bayes’ Theorem calculations using the IMDb dataset.  |
| Natinael Borana | Part 3 and 4 | Completed all Gradient Descent code and plots, manually calculated iteration 2, and corrected errors in the overall manual calculations. |
| Maxime Kwizera | Part 2 and 3 | Completed Gradient Descent iteration 3 calculations and Bayes’ code and report. |

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
