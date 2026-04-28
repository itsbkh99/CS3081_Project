# Feature Selection AI Agent
> CS3081 · Artificial Intelligence · Effat University · Spring 2026  
> **Team:** Faisal Yahya (S23208857) · Faisal Shamsi · Omar Almutairi  
> **Instructor:** Dr. Naila Marir

An intelligent agent that automatically finds the **minimum set of features** needed for **maximum classification accuracy**, using A\* Search guided by a Mutual Information heuristic.

---

## What Does It Do?

Given any dataset with many columns, the agent asks:

> *"Do I really need all 30 columns to predict accurately, or can I get the same (or better) result with just 6?"*

**Example result on Breast Cancer dataset:**
- Started with **30 features**
- Agent selected **6 features**
- Accuracy improved from **92.9% → 95.6%**
- **80% dimensionality reduction**
- **0 fake/noise features selected** (correctly ignored all injected garbage)

---

## Project Structure

```
CS3081_Project/
├── agent.py          # Main entry point — CLI interface
├── app.py            # Flask web server — browser UI
├── environment.py    # Dataset loader + adversarial injection
├── evaluator.py      # Naive Bayes scorer + knowledge base cache
├── heuristics.py     # Mutual Information heuristic for A*
├── search.py         # A* Search + Forward Selection algorithms
├── static/
│   └── index.html    # Web UI frontend
└── requirements.txt  # Python dependencies
```

---

## Installation

### 1. Install dependencies (one time only)

```bash
pip3 install scikit-learn numpy pandas matplotlib seaborn scipy flask
```

---

## Option A — Web UI (Recommended)

No terminal commands needed after setup. Just drag, drop, and click.

### Start the server

```bash
python3 app.py
```

You should see:
```
🚀  Feature Selection AI Agent — Web UI
Open your browser at: http://localhost:5000
```

### Open in browser

Go to:
```
http://127.0.0.1:5000
```

### How to use

1. **Drop any CSV file** into the upload area (any Kaggle dataset works)
2. **Choose algorithm** — A\* Search (optimal) or Forward Selection (greedy baseline)
3. **Adjust sliders** (optional):
   - Gaussian Noise σ — adds noise to test robustness
   - Extra Fake Features — injects random columns to test if agent ignores them
4. **Leave Target Column blank** — it auto-detects which column to predict
5. **Click Run Agent**
6. See results: accuracy, features selected, MI bar chart

### Stop the server

Press `Ctrl + C` in the terminal.

---

## Option B — Terminal (Command Line)

### Basic runs

```bash
# Default (breast cancer dataset, A* search)
python3 agent.py

# Choose a built-in dataset
python3 agent.py --dataset breast_cancer
python3 agent.py --dataset wine
python3 agent.py --dataset ionosphere

# Use any Kaggle CSV file (drop it in the folder first)
python3 agent.py --dataset titanic.csv
python3 agent.py --dataset heart.csv
python3 agent.py --dataset world_happiness_2026.csv

# Choose algorithm
python3 agent.py --search astar       # A* (default, optimal)
python3 agent.py --search forward     # Greedy forward selection (baseline)
```

### Adversarial tests

```bash
# Add Gaussian noise to features
python3 agent.py --noise 0.3

# Inject 50% extra fake/random features
python3 agent.py --extra 0.5

# Flip 10% of labels
python3 agent.py --flip 0.1

# All adversarial at once
python3 agent.py --noise 0.3 --extra 0.5 --flip 0.1
```

### Best comparison (for report/presentation)

```bash
# A* vs Forward Selection under same adversarial conditions
python3 agent.py --dataset breast_cancer --search astar --noise 0.3 --extra 0.5
python3 agent.py --dataset breast_cancer --search forward --noise 0.3 --extra 0.5
```

### Run full experiment suite

```bash
# Runs all datasets × both algorithms × noise levels → saves CSV report
python3 agent.py --experiment
```

Results are saved to `results/experiment_results.csv`.

---

## Using Custom Kaggle Datasets

1. Download any CSV from [kaggle.com](https://kaggle.com)
2. Drop it into the `CS3081_Project` folder
3. Run it:

```bash
# Target column is auto-detected
python3 agent.py --dataset your_file.csv

# Or specify the target column manually
python3 agent.py --dataset your_file.csv --target ColumnName
```

**The auto-detector works by:**
1. Looking for columns named `target`, `label`, `survived`, `outcome`, `score`, etc.
2. Finding binary columns (0/1, yes/no)
3. Finding the column with fewest unique values
4. Falling back to the last column

**Compatible with any CSV that has:**
- At least 100 rows
- At least 5 numeric feature columns
- A classification target (category prediction, not number prediction)

---

## AI Techniques Used

| Component | Technique | Purpose |
|---|---|---|
| `search.py` | **A\* Search** | Finds optimal feature subset intelligently |
| `heuristics.py` | **Mutual Information** | Admissible heuristic — guides A\* toward useful features |
| `evaluator.py` | **Naive Bayes + 5-fold CV** | Scores each candidate feature subset |
| `evaluator.py` | **Knowledge Base Cache** | Avoids re-evaluating already-seen subsets |
| `environment.py` | **Adversarial Injection** | Tests robustness with noise, fake features, label flips |

### Why A\* is optimal

The MI heuristic `h(n)` is **admissible** — it never overestimates the remaining accuracy gain. This guarantees A\* returns the globally optimal feature subset (not just a locally good one like greedy methods).

```
f(n) = g(n) + h(n)
  g(n) = −current_accuracy     (cost so far)
  h(n) = −MI heuristic         (estimated remaining gain)
```

---

## Adversarial Challenges Addressed

| Challenge | How We Test It | Flag |
|---|---|---|
| **Noisy Observations** | Gaussian noise added to all features | `--noise 0.3` |
| **Deceptive Signals** | Random fake columns injected | `--extra 0.5` |
| **Incomplete Knowledge** | Labels randomly flipped | `--flip 0.1` |

---

## Results Summary

| Dataset | Features | A\* Accuracy | Forward Accuracy | Baseline | Reduction |
|---|---|---|---|---|---|
| Breast Cancer | 30 → 6 | **95.6%** | 93.1% | 92.9% | 80% |
| Wine | 13 → 4 | **97.2%** | 95.8% | 94.4% | 69% |
| Ionosphere | 34 → 8 | **91.4%** | 88.7% | 87.2% | 76% |

---

## Requirements

```
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.12.0
flask>=3.0.0
```

---

## Domain

**Decision Support** — the agent helps decide which features (columns) in a dataset actually matter for making accurate predictions, reducing complexity and improving performance automatically.
