"""
app.py
------
Flask web server for the Feature Selection AI Agent.
Run with: python3 app.py
Then open: http://localhost:5000

CS3081: Artificial Intelligence · Effat University · Spring 2026
Team: Faisal Yahya (S23208857), Faisal Shamsi, Omar Almutairi
"""

import os
import sys
import json
import time
import threading
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))
from environment import load_csv_dataset, set_target
from evaluator import NaiveBayesEvaluator
from heuristics import compute_mi_scores
from search import forward_selection, astar_search

app = Flask(__name__, static_folder="static")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Store job results in memory ──
jobs = {}


def run_job(job_id, filepath, algorithm, noise, extra, target_col):
    """Run the feature selection experiment in a background thread."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["log"] = []

        def log(msg):
            jobs[job_id]["log"].append(msg)

        # Load dataset
        if target_col:
            set_target(target_col)
        else:
            set_target(None)

        X_raw, y, feat_names = load_csv_dataset(filepath, target_col=target_col or None)
        # Force all features to numeric, drop anything that can't be converted
        import pandas as pd
        df_check = pd.DataFrame(X_raw, columns=feat_names)
        df_check = df_check.apply(pd.to_numeric, errors='coerce')
        df_check = df_check.dropna(axis=1)
        X_raw = df_check.values
        feat_names = list(df_check.columns)
        n_original = X_raw.shape[1]
        n_samples  = X_raw.shape[0]

        log(f"Loaded {n_samples} samples × {n_original} features")
        log(f"Running {algorithm.upper()} search...")

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # Inject noise if requested
        rng = np.random.default_rng(42)
        if noise > 0:
            X = X + rng.normal(0, noise, size=X.shape)
            log(f"Injected Gaussian noise (σ={noise})")

        n_extra = int(n_original * extra)
        if n_extra > 0:
            X_extra = rng.standard_normal((n_samples, n_extra))
            X = np.hstack([X, X_extra])
            feat_names = feat_names + [f"noise_feat_{i}" for i in range(n_extra)]
            log(f"Injected {n_extra} fake features")

        n_total = X.shape[1]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Baseline
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        base_acc = accuracy_score(y_test, clf.predict(X_test))
        log(f"Baseline (all features): {base_acc:.4f} accuracy")

        # MI scores
        mi_scores = compute_mi_scores(X_train, y_train, seed=42)
        evaluator = NaiveBayesEvaluator(X_train, y_train, cv_folds=5, seed=42)

        t0 = time.time()
        if algorithm == "forward":
            result = forward_selection(evaluator, n_total)
        else:
            result = astar_search(evaluator, n_total, mi_scores)
        search_time = round(time.time() - t0, 2)

        # Final evaluation
        mask = np.zeros(n_total, dtype=bool)
        mask[result.selected_features] = True
        clf2 = GaussianNB()
        clf2.fit(X_train[:, mask], y_train)
        final_acc = accuracy_score(y_test, clf2.predict(X_test[:, mask]))
        final_f1  = f1_score(y_test, clf2.predict(X_test[:, mask]), average="macro")

        orig_selected  = [f for f in result.selected_features if f < n_original]
        noise_selected = [f for f in result.selected_features if f >= n_original]
        selected_names = [feat_names[i] for i in sorted(result.selected_features) if i < len(feat_names)]
        dim_reduction  = round(100 * (1 - len(result.selected_features) / n_total), 1)

        log(f"Search complete in {search_time}s")
        log(f"Selected {len(result.selected_features)} / {n_total} features")

        jobs[job_id].update({
            "status": "done",
            "baseline_accuracy":    round(base_acc, 4),
            "agent_accuracy":       round(final_acc, 4),
            "accuracy_delta":       round(final_acc - base_acc, 4),
            "f1_score":             round(final_f1, 4),
            "n_selected":           len(result.selected_features),
            "n_total":              n_total,
            "n_original":           n_original,
            "n_orig_selected":      len(orig_selected),
            "n_noise_selected":     len(noise_selected),
            "dim_reduction":        dim_reduction,
            "subsets_evaluated":    result.n_subsets_evaluated,
            "search_time":          search_time,
            "selected_features":    selected_names,
            "algorithm":            algorithm.upper(),
            "mi_scores":            [(feat_names[i], round(float(mi_scores[i]), 4))
                                     for i in np.argsort(mi_scores)[::-1][:15]
                                     if i < len(feat_names)],
        })

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/run", methods=["POST"])
def run():
    f         = request.files.get("file")
    algorithm = request.form.get("algorithm", "astar")
    noise     = float(request.form.get("noise", 0))
    extra     = float(request.form.get("extra", 0))
    target    = request.form.get("target", "").strip() or None

    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(filepath)

    job_id = str(int(time.time() * 1000))
    jobs[job_id] = {"status": "queued", "log": []}

    thread = threading.Thread(target=run_job,
                               args=(job_id, filepath, algorithm, noise, extra, target))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    return jsonify(jobs.get(job_id, {"status": "not_found"}))


if __name__ == "__main__":
    print("\n  🚀  Feature Selection AI Agent — Web UI")
    print("  Open your browser at: http://localhost:5000\n")
    app.run(debug=False, port=5000)
