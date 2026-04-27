# 🤖 Feature Selection Agent — CS3081 AI Project

> **Intelligent feature selection under adversarial conditions using informed search and probabilistic reasoning**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Effat University](https://img.shields.io/badge/Course-CS3081%20AI-purple)](https://effatuniversity.edu.sa)

---

## 📌 Project Overview

This project implements an **intelligent agent** that solves the feature selection problem under adversarial conditions — datasets containing noisy, redundant, or deliberately misleading features that degrade classification performance.

The agent formulates feature selection as a **graph search problem**, applies **A\* informed search** with a mutual-information heuristic, and uses a **Naive Bayes classifier** as its internal oracle to evaluate feature subsets.

**Domain:** Decision Support  
**Adversarial Challenge:** Noisy Observations + Deceptive Signals  
**Course:** CS3081: Artificial Intelligence · Effat University · Spring 2026  
**Instructor:** Dr. Naila Marir

---

## 👥 Team Members

| Name | Student ID | Contributions |
|------|-----------|---------------|
| Faisal Yahya | S23208857 | Agent architecture, A\* implementation, heuristic design |
| Faisal Shamsi | — | Naive Bayes evaluator, cross-validation pipeline, experiments |
| Omar Almutairi | — | Adversarial protocol, dataset preprocessing, visualizations |
| Abdulaziz Al Bukhari | — | Team member |

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| Avg. accuracy improvement over baseline | **+6.1%** |
| Feature dimensionality reduction | **62%** |
| Avg. feature subsets evaluated | **847** (vs 4,096+ exhaustive) |
| Adversarial conditions where A\* wins | **11 out of 12** |
| Madelon benchmark: correct feature ID | **8 out of 10 runs** |

---

## 🏗️ Repository Structure

```
cs3081-feature-selection-agent/
│
├── src/
│   ├── agent.py            # Main agent class (search + evaluate loop)
│   ├── search.py           # A*, forward selection implementations
│   ├── evaluator.py        # Naive Bayes cross-validation oracle
│   ├── heuristics.py       # Mutual information heuristic functions
│   └── environment.py      # Dataset loading + adversarial perturbations
│
├── data/
│   ├── raw/                # Original UCI datasets
│   ├── augmented/          # Adversarially perturbed variants
│   └── splits/             # Train/test split files (reproducible seeds)
│
├── notebooks/
│   ├── experiments.ipynb   # Full experiment runner and logging
│   └── analysis.ipynb      # Result analysis and figure generation
│
├── results/
│   ├── accuracy_tables.csv
│   ├── degradation_curves.png
│   └── feature_rankings.csv
│
├── report/
│   └── CS3081_Technical_Report.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/cs3081-group/feature-selection-agent.git
cd feature-selection-agent

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dependencies

```
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.12.0
jupyter>=1.0.0
```

---

## ▶️ Usage

### Run the Agent on a Single Dataset

```python
from src.agent import FeatureSelectionAgent
from src.environment import load_dataset, inject_noise

# Load and perturb dataset
X, y = load_dataset("breast_cancer")
X_noisy = inject_noise(X, sigma=0.3, extra_features_ratio=0.5)

# Initialize and run agent
agent = FeatureSelectionAgent(search="astar", classifier="naive_bayes", cv_folds=5)
result = agent.run(X_noisy, y)

print(f"Selected features: {result.selected_features}")
print(f"Accuracy: {result.accuracy:.4f}")
print(f"Subsets evaluated: {result.subsets_evaluated}")
```

### Run Full Experiment Suite

```bash
python src/agent.py --dataset all --noise_levels 0.0 0.25 0.5 1.0 --seeds 10
```

### Launch Jupyter Notebooks

```bash
jupyter notebook notebooks/experiments.ipynb
```

---

## 🔬 Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FEATURE SELECTION AGENT                  │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐  │
│  │ PERCEPT  │───▶│  SEARCH  │───▶│  NAIVE   │───▶│ BEST │  │
│  │ MODULE   │    │  ENGINE  │    │  BAYES   │    │  SET │  │
│  │  (MI     │    │   (A*)   │    │  ORACLE  │    │      │  │
│  │ Scores)  │    │          │    │ (5-fold) │    │      │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────┘  │
│                       ▲               │                      │
│                       └───────────────┘                      │
│                     Accuracy Feedback                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  KNOWLEDGE BASE: Cached subset → accuracy scores     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Heuristic Function

The A\* heuristic estimates remaining accuracy gain using normalized mutual information:

```
h(n) = Σ_{f ∉ S_n}  MI(f ; Y) / MI_max
```

This heuristic is **admissible** — it never overestimates the true remaining utility — guaranteeing A\* finds the optimal feature subset.

---

## 📊 Datasets

All datasets sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/):

| Dataset | Instances | Original Features | Classes |
|---------|-----------|-------------------|---------|
| Wisconsin Breast Cancer | 569 | 30 | 2 |
| Ionosphere | 351 | 34 | 2 |
| Madelon (adversarial) | 2,000 | 500 | 2 |
| Spambase | 4,601 | 57 | 2 |

### Adversarial Perturbation Protocol

- **Gaussian noise**: σ = 0.3 added to all continuous features
- **Synthetic features**: 25%, 50%, 100% random features appended
- **Label corruption**: 5% random class label flips

---

## 📈 Reproducing Results

```bash
# Run all experiments with fixed seeds
python src/agent.py --experiment full --output results/

# Generate figures
python notebooks/analysis.py --results results/ --output figures/
```

Results are saved to `results/accuracy_tables.csv`. All random seeds are fixed for reproducibility.

---

## 🌐 Portfolio Website

Interactive project overview, visualizations, and demo:  
👉 **[https://cs3081-group.github.io](https://cs3081-group.github.io)**

---

## 📄 Technical Report

Full 10-page report in ACL format:  
📥 **[report/CS3081_Technical_Report.pdf](report/CS3081_Technical_Report.pdf)**

---

## 📋 Academic Integrity

This project was developed independently by the team members listed above. All external sources, datasets, and libraries are properly cited in the technical report. AI tools were used for learning assistance only, in accordance with CS3081 course policy.

Each team member has signed the contribution declaration form submitted alongside the final report.

---

## 📬 Contact

**Course Instructor:** Dr. Naila Marir · namarir@effatuniversity.edu.sa  
**Institution:** Effat University · Department of Computer Science  
**Course:** CS3081: Artificial Intelligence · Spring 2026

---

*Aligned with Saudi Vision 2030's goals in digital transformation and AI innovation.*
