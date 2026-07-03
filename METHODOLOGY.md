# FIFA World Cup Match Outcome Prediction Using Logistic Regression Implemented from First Principles

**Author:** Bimarsh Shrestha
**Program:** Master of Data Science, Macquarie University
**Project Type:** Independent Portfolio Project
**Date:** July 2026

---

## Results actually achieved

This document was originally written as a project plan before training.
The sections below (theory, pipeline design, implementation) describe the
plan as executed. Here are the final numbers from the completed run, for
reference before diving into the derivation:

- **Final dataset:** 1,129 team-match rows (FIFA World Cup matches with
  available FIFA ranking history)
- **Held-out test set:** accuracy 0.724, precision 0.712, recall 0.483,
  F1 0.575, ROC-AUC 0.788
- **5-fold cross-validated AUC:** 0.730 ± 0.040
- **Validated against scikit-learn:** weights differ by at most ~0.02 per
  feature; the from-scratch and library implementations converge to the
  same solution
- **Dominant feature:** FIFA rank differential (weight ≈ 0.77), roughly
  4&times; the magnitude of any other feature
- **L2 regularisation:** tested across &lambda; &isin; [0, 10], negligible
  effect — consistent with a low feature-to-sample ratio rather than
  overfitting

---



## Abstract

This project develops a binary classification model to predict the outcome of FIFA World Cup matches (win vs. not-win) using logistic regression implemented entirely from first principles in Python with NumPy — no machine learning libraries such as scikit-learn or tidymodels are used for the core model. The purpose is twofold: (1) to demonstrate a rigorous understanding of the mathematical foundations of logistic regression, including maximum likelihood estimation, the cross-entropy loss function, and gradient descent optimisation; and (2) to produce a complete, reproducible machine learning pipeline covering data acquisition, cleaning, feature engineering, model training, evaluation, and interpretation. The model is benchmarked using accuracy, precision, recall, F1-score, ROC-AUC, and log loss under k-fold cross-validation, and model coefficients are interpreted through odds ratios to identify which match statistics most strongly influence winning probability.

---

## 1. Introduction

### 1.1 Motivation

Predicting the outcome of football matches is a classic applied statistics problem: outcomes are binary (or categorical), data is noisy, and the relationships between team statistics and results are probabilistic rather than deterministic. Logistic regression is the natural starting point for this problem because it directly models the *probability* of an outcome rather than the outcome itself.

Implementing the model from scratch — rather than calling `sklearn.linear_model.LogisticRegression` — forces engagement with every mathematical component: the linear predictor, the sigmoid transformation, the likelihood function, the loss surface, and the optimisation procedure. This depth of understanding is precisely what distinguishes a data scientist from a library user.

### 1.2 Problem Statement

Given historical World Cup match data with team-level features (e.g., FIFA ranking difference, recent form, goals scored, possession statistics), predict the probability that a given team wins the match.

Formally: given a feature vector $\mathbf{x} \in \mathbb{R}^d$ describing a match, estimate

$$P(y = 1 \mid \mathbf{x})$$

where $y = 1$ denotes a win and $y = 0$ denotes a draw or loss.

### 1.3 Objectives

1. Derive and implement logistic regression using only NumPy.
2. Build a full, reproducible ML pipeline (data → features → training → evaluation).
3. Derive the gradient of the cross-entropy loss analytically and implement batch gradient descent.
4. Evaluate the model rigorously with cross-validation and multiple metrics.
5. Interpret the fitted model via coefficients and odds ratios.

---

## 2. Theoretical Background

### 2.1 Why Not Linear Regression?

Linear regression models a continuous response:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

Its output is unbounded ($-\infty$ to $+\infty$), so it cannot be interpreted as a probability. Applying it to a 0/1 outcome also violates its core assumptions (normally distributed residuals, constant variance). Logistic regression fixes this by passing the linear predictor through a squashing function.

### 2.2 The Linear Predictor

The model first computes a raw score, often written $z$:

$$z = \mathbf{w}^\top \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$$

- $x_j$: the $j$-th **feature** (e.g., ranking difference, average goals in last 5 matches)
- $w_j$: the **weight** for feature $j$ — a learned multiplier expressing how much that feature influences the outcome, and in which direction
- $b$: the **bias/intercept** — the baseline log-odds when all features are zero

The weights are *not* part of the data. They are the model's learnable parameters, initialised (near zero or randomly) and then optimised.

### 2.3 The Sigmoid (Logistic) Function

The sigmoid maps any real number to the interval $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Key properties:

- $\sigma(0) = 0.5$ (maximum uncertainty)
- $z \to +\infty \Rightarrow \sigma(z) \to 1$
- $z \to -\infty \Rightarrow \sigma(z) \to 0$
- S-shaped, smooth, and differentiable everywhere
- Its derivative has an elegant closed form used in the gradient derivation:

$$\sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big)$$

The model's predicted probability is therefore:

$$\hat{p} = P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$$

### 2.4 Odds and Log-Odds Interpretation

Logistic regression is a *linear model in log-odds space*:

$$\log\frac{\hat{p}}{1 - \hat{p}} = \mathbf{w}^\top \mathbf{x} + b$$

The quantity $\hat{p}/(1-\hat{p})$ is the **odds** of winning. Exponentiating a weight gives an **odds ratio**: $e^{w_j}$ is the multiplicative change in the odds of winning for a one-unit increase in feature $j$, holding other features constant. This is what makes logistic regression highly interpretable — critical for the interpretability section (Section 8).

### 2.5 Maximum Likelihood Estimation (Where the Loss Comes From)

Each match outcome $y_i \in \{0, 1\}$ is modelled as a Bernoulli trial with success probability $\hat{p}_i$. The likelihood of the full dataset of $n$ independent matches is:

$$L(\mathbf{w}, b) = \prod_{i=1}^{n} \hat{p}_i^{\,y_i} (1 - \hat{p}_i)^{1 - y_i}$$

Maximising $L$ is equivalent to minimising the **negative log-likelihood**, which is exactly the **binary cross-entropy loss**:

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^{n} \Big[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \Big]$$

### 2.6 Why Cross-Entropy and Not Squared Error?

1. **Statistical justification:** cross-entropy is the negative log-likelihood of the Bernoulli model — it is the *principled* loss, not an arbitrary choice.
2. **Penalty behaviour:** if the model predicts $\hat{p} = 0.01$ for a match the team actually won, the loss term is $-\log(0.01) \approx 4.6$ — a huge penalty. Squared error would give only $(1 - 0.01)^2 \approx 0.98$. Cross-entropy punishes *confident wrong* predictions severely.
3. **Optimisation geometry:** with the sigmoid, cross-entropy yields a **convex** loss surface with a single global minimum. Squared error with a sigmoid is non-convex and can trap gradient descent in flat regions.

### 2.7 Gradient Derivation (Full Working)

We need $\partial \mathcal{L} / \partial w_j$ and $\partial \mathcal{L} / \partial b$. Using the chain rule on a single observation with $\hat{p} = \sigma(z)$ and $z = \mathbf{w}^\top\mathbf{x} + b$:

**Step 1 — derivative of loss w.r.t. $\hat{p}$:**

$$\frac{\partial \mathcal{L}}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1 - y}{1 - \hat{p}}$$

**Step 2 — derivative of $\hat{p}$ w.r.t. $z$ (sigmoid derivative):**

$$\frac{\partial \hat{p}}{\partial z} = \hat{p}(1 - \hat{p})$$

**Step 3 — derivative of $z$ w.r.t. $w_j$:**

$$\frac{\partial z}{\partial w_j} = x_j$$

**Multiply the chain together** — the algebra collapses beautifully:

$$\frac{\partial \mathcal{L}}{\partial w_j} = (\hat{p} - y)\, x_j$$

Averaged over all $n$ observations, in vectorised form:

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{n} \mathbf{X}^\top (\hat{\mathbf{p}} - \mathbf{y}), \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)$$

The gradient is simply *(prediction error) × (feature value)* — one of the most elegant results in machine learning.

### 2.8 Gradient Descent

Starting from initial weights $\mathbf{w}^{(0)}$, repeat until convergence:

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \, \nabla_{\mathbf{w}} \mathcal{L}, \qquad b^{(t+1)} = b^{(t)} - \eta \, \frac{\partial \mathcal{L}}{\partial b}$$

where $\eta$ is the **learning rate**:

- $\eta$ too large → the loss oscillates or diverges (overshooting the minimum)
- $\eta$ too small → convergence is painfully slow
- Typical starting values: 0.01–0.1 for standardised features

Because the cross-entropy loss for logistic regression is convex, batch gradient descent is guaranteed to converge to the global minimum for a suitable learning rate.

**Convergence criteria:** stop when the change in loss between iterations falls below a tolerance (e.g., $10^{-6}$), or after a fixed maximum number of iterations (e.g., 10,000).

### 2.9 Regularisation (Extension)

To prevent overfitting, an L2 penalty can be added:

$$\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2n} \|\mathbf{w}\|^2$$

which adds $\frac{\lambda}{n} w_j$ to each weight gradient. The bias term is conventionally not regularised. This connects directly to the ridge/lasso material from COMP8107.

---

## 3. Data and Pipeline Architecture

### 3.1 Pipeline Overview

```
Raw data (CSV)
   │
   ▼
1. Data loading & validation
   │
   ▼
2. Cleaning (missing values, duplicates, type coercion)
   │
   ▼
3. Exploratory Data Analysis (EDA)
   │
   ▼
4. Feature engineering
   │
   ▼
5. Train/test split (80/20, stratified)
   │
   ▼
6. Feature scaling (fit on train ONLY)
   │
   ▼
7. Model training (from-scratch logistic regression)
   │
   ▼
8. Evaluation (k-fold CV + held-out test set)
   │
   ▼
9. Interpretation (coefficients, odds ratios)
```

### 3.2 Data Sources

Suitable public datasets include:

- International football results 1872–present (Kaggle, `martj42/international-football-results-from-1872-to-2017`)
- FIFA World Rankings (Kaggle)
- FIFA World Cup match statistics datasets (goals, possession, shots)

### 3.3 Data Cleaning Rules

- Drop exact duplicate rows.
- Handle missing values explicitly: document whether rows are dropped or imputed (median imputation for numeric features), and justify the choice.
- Standardise team names across datasets (e.g., "Korea Republic" vs "South Korea") before joining ranking data to match data.
- Restrict to a consistent era if rules/format changed (e.g., post-1994, 3 points per win).

### 3.4 Preventing Data Leakage

**Critical principle:** every feature must be computable *before kick-off*. Features such as goals scored *in the match being predicted* leak the outcome. Use only pre-match information: rankings, historical form, head-to-head record. Similarly, scaling parameters (mean, standard deviation) must be computed on the training set only and applied to the test set.

---

## 4. Feature Engineering

Candidate pre-match features (all computed per team, per match):

| Feature | Description | Expected direction |
|---|---|---|
| `rank_diff` | FIFA ranking of opponent minus own ranking | Positive → higher win probability |
| `form_5` | Points per game in last 5 matches | Positive |
| `goals_avg_5` | Average goals scored in last 5 matches | Positive |
| `conceded_avg_5` | Average goals conceded in last 5 matches | Negative |
| `h2h_winrate` | Historical win rate vs this opponent | Positive |
| `is_host` | 1 if playing as host nation | Positive (home advantage) |
| `days_rest` | Days since previous match | Positive (small) |
| `confed_strength` | Confederation coefficient (UEFA/CONMEBOL vs others) | Positive |

**Feature scaling:** standardise each numeric feature ($z$-score) so gradient descent converges quickly and weights are comparable in magnitude:

$$x_j' = \frac{x_j - \mu_j}{\sigma_j}$$

---

## 5. From-Scratch Implementation (NumPy Only)

### 5.1 Core Model Class

```python
import numpy as np

class LogisticRegressionScratch:
    """Binary logistic regression trained with batch gradient descent.

    Implements the model from first principles:
    - sigmoid activation
    - binary cross-entropy loss (negative log-likelihood)
    - analytically derived gradients
    - optional L2 regularisation
    """

    def __init__(self, learning_rate=0.05, n_iterations=10_000,
                 tol=1e-6, l2_lambda=0.0, verbose=False):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.l2_lambda = l2_lambda
        self.verbose = verbose
        self.w = None
        self.b = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        # Clip z for numerical stability (prevents overflow in exp)
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _loss(self, y, p):
        # Clip p away from exact 0/1 so log() never returns -inf
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = (self.l2_lambda / (2 * len(y))) * np.sum(self.w ** 2)
        return ce + reg

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        prev_loss = np.inf
        for i in range(self.n_iterations):
            # Forward pass
            z = X @ self.w + self.b
            p = self._sigmoid(z)

            # Gradients: (1/n) * X^T (p - y), plus L2 term
            error = p - y
            grad_w = (X.T @ error) / n_samples + (self.l2_lambda / n_samples) * self.w
            grad_b = np.mean(error)

            # Parameter update
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            # Track loss and check convergence
            loss = self._loss(y, p)
            self.loss_history.append(loss)
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}, loss={loss:.6f}")
                break
            prev_loss = loss
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

### 5.2 Supporting Utilities (Also From Scratch)

```python
def standard_scale(X_train, X_test):
    """Fit scaling on train only to avoid data leakage."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0          # guard against constant columns
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def train_test_split_stratified(X, y, test_size=0.2, seed=42):
    """Stratified split preserving class balance."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_pos_test = int(len(idx_pos) * test_size)
    n_neg_test = int(len(idx_neg) * test_size)
    test_idx = np.concatenate([idx_pos[:n_pos_test], idx_neg[:n_neg_test]])
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

### 5.3 Evaluation Metrics From Scratch

```python
def confusion_counts(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp, _, fp, _ = confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    tp, _, _, fn = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def roc_auc(y_true, y_scores):
    """AUC via the rank-sum (Mann-Whitney U) formulation."""
    order = np.argsort(y_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    sum_pos_ranks = np.sum(ranks[y_true == 1])
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

### 5.4 K-Fold Cross-Validation From Scratch

```python
def k_fold_cv(X, y, k=5, seed=42, **model_kwargs):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    results = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale inside the fold — never before splitting
        X_tr_s, X_val_s, _, _ = standard_scale(X_tr, X_val)

        model = LogisticRegressionScratch(**model_kwargs).fit(X_tr_s, y_tr)
        proba = model.predict_proba(X_val_s)
        pred = (proba >= 0.5).astype(int)

        results.append({
            "fold": i + 1,
            "accuracy": accuracy(y_val, pred),
            "precision": precision(y_val, pred),
            "recall": recall(y_val, pred),
            "f1": f1_score(y_val, pred),
            "auc": roc_auc(y_val, proba),
        })
    return results
```

---

## 6. Experimental Protocol

1. **Split:** 80% train / 20% held-out test, stratified by outcome.
2. **Model selection:** 5-fold cross-validation *on the training set only* to tune learning rate $\eta \in \{0.01, 0.05, 0.1\}$ and L2 strength $\lambda \in \{0, 0.01, 0.1, 1\}$.
3. **Final training:** refit the best configuration on the full training set.
4. **Final evaluation:** report all metrics once on the untouched test set.
5. **Baselines for comparison:**
   - Majority-class classifier (predicts the most common outcome)
   - "Higher-ranked team always wins" rule-based baseline
   - The model must beat both to demonstrate genuine learning.

**Diagnostics to include:**
- Loss curve (loss vs iteration) — demonstrates gradient descent converging.
- Learning-rate comparison plot — visual proof of the overshoot/slow-convergence trade-off.
- Calibration check — group predictions into probability bins and compare predicted vs actual win rates.

---

## 7. Evaluation Metrics: Definitions and Rationale

| Metric | Formula | Why it matters here |
|---|---|---|
| Accuracy | $(TP+TN)/n$ | Overall correctness; misleading if classes are imbalanced (wins are a minority when draws exist) |
| Precision | $TP/(TP+FP)$ | Of matches predicted as wins, how many were wins |
| Recall | $TP/(TP+FN)$ | Of actual wins, how many the model caught |
| F1-score | Harmonic mean of P and R | Single balanced number for imbalanced data |
| ROC-AUC | Rank-based | Threshold-independent ranking quality of predicted probabilities |
| Log loss | Cross-entropy on test set | Directly measures probability quality, not just labels |

Report the confusion matrix explicitly, and report cross-validation results as **mean ± standard deviation** across folds to communicate variance.

---

## 8. Model Interpretation

After training on standardised features:

1. **Weight magnitudes** rank feature importance (valid because features are on the same scale).
2. **Weight signs** confirm or contradict football intuition (e.g., `rank_diff` should be positive).
3. **Odds ratios:** for feature $j$, $e^{w_j}$ = multiplicative change in win odds per one standard deviation increase.

Example interpretation template:

> "A one-standard-deviation improvement in recent form (`form_5`) multiplies the odds of winning by $e^{0.62} \approx 1.86$ — an 86% increase in win odds, holding all other features constant."

This section is what interviewers care about most: it shows the model produces *insight*, not just predictions.

---

## 9. Extensions for a Master's-Level Portfolio

To elevate this beyond a basic implementation:

1. **Model comparison:** benchmark the from-scratch model against scikit-learn's `LogisticRegression` (to validate correctness — coefficients should closely match) and against a random forest / gradient boosting model (to quantify the linearity gap).
2. **Multinomial extension:** extend to three classes (win/draw/loss) using softmax regression — a natural generalisation of the sigmoid.
3. **Probability calibration analysis:** reliability diagrams showing whether a predicted 70% really wins ~70% of the time.
4. **Tournament simulation:** use the fitted model to Monte-Carlo simulate an entire World Cup bracket and produce championship probabilities per team — an eye-catching deliverable.
5. **Bayesian logistic regression:** place priors on the weights and sample the posterior with Metropolis-Hastings — directly leveraging the STAT8150 material and differentiating the project substantially.
6. **Deployment:** wrap the final model in a small Streamlit or Flask app that takes two teams and outputs a win probability.

---

## 10. Suggested Repository Structure

```
worldcup-logreg/
├── README.md                  # Project summary, results, how to run
├── data/
│   ├── raw/                   # Original downloaded CSVs (never edited)
│   └── processed/             # Cleaned, feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory analysis
│   └── 02_results.ipynb       # Final results and plots
├── src/
│   ├── data_prep.py           # Loading, cleaning, feature engineering
│   ├── model.py               # LogisticRegressionScratch class
│   ├── metrics.py             # From-scratch evaluation metrics
│   ├── cross_validation.py    # K-fold CV implementation
│   └── train.py               # End-to-end training script
├── reports/
│   └── figures/               # Loss curves, ROC curves, calibration plots
└── requirements.txt           # numpy, pandas, matplotlib only
```

---

## 11. Limitations

- Logistic regression assumes a **linear relationship in log-odds space**; genuine feature interactions (e.g., form mattering more against strong opponents) require explicit interaction terms.
- World Cup matches are relatively rare events; the training set is small, so variance across CV folds may be substantial.
- Football outcomes have high irreducible noise — even bookmakers' models achieve modest accuracy. Framing expectations honestly (e.g., beating baselines by a clear margin rather than achieving 90% accuracy) demonstrates statistical maturity.
- Draws complicate the binary framing; the multinomial extension (Section 9.2) addresses this properly.

---

## 12. Conclusion

This project demonstrates end-to-end machine learning competence: mathematical derivation (MLE → cross-entropy → analytic gradients), correct implementation from first principles, disciplined experimental methodology (leakage-free pipeline, cross-validation, baseline comparison), and interpretable communication of results through odds ratios. The extensions in Section 9 — particularly the Bayesian variant and tournament simulation — provide clear pathways to differentiate the work at a Master's portfolio level.

---

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. — Chapter 4 (Linear Methods for Classification).
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. — Chapter 4.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. — Section 4.3 (Probabilistic Discriminative Models).
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. — Chapter 4 (Numerical Computation, gradient-based optimisation).
5. Kaggle: International football results from 1872 to 2025. https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
