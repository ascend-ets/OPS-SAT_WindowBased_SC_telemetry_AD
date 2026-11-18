#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, average_precision_score, confusion_matrix
from pyod.utils.data import precision_n_scores

def evaluate_metrics(y_test, y_pred, y_proba=None, digits=3):
    if (y_test == 1).any():
        res = {"Accuracy": round(accuracy_score(y_test, y_pred), digits),
                "Precision": round(precision_score(y_test, y_pred, labels=[0,1], zero_division=0), digits),
                "Recall": round(recall_score(y_test, y_pred, labels=[0,1], zero_division=0), digits),
                "F1": round(f1_score(y_test, y_pred, labels=[0,1], zero_division=0), digits),
                "MCC": float(round(matthews_corrcoef(y_test, y_pred), ndigits=digits))}
        if y_proba is not None:
            res["AUC_PR"] = float(average_precision_score(y_test, y_proba).round(digits))
            res["AUC_ROC"] = float(roc_auc_score(y_test, y_proba).round(digits))
            res["PREC_N_SCORES"] = round(precision_n_scores(y_test, y_proba), digits)
    else:
        res = {"Accuracy": round(accuracy_score(y_test, y_pred), digits)}
    return res

window_sizes = [15, 30, 45, 60, 75, 80]

models = ['ADA', 'RF']
features = ['I', 'II', 'III']

metrics = ["Precision", "Recall", "F1"]

model_names = {
    "ADA": "AdaBoost",
    "RF": "Random Forest"
}

results = {}

for model in models:
    model_name = model_names[model]
    results[model_name] = {}
    for feat in features:
        results[model_name][feat] = {metric: [] for metric in metrics}
        for w in window_sizes:
            df = pd.read_csv(f'experiments/w{w}_s1_lstm0.3/classifier/{feat}/predictions_{model}.csv',)
            y_true = df['y_true'].values
            y_pred = df['y_pred'].values
            scores = evaluate_metrics(y_true, y_pred)
            
            for metric in metrics:
                results[model_name][feat][metric].append(scores[metric])
#%%

# ==========================================
# CONFIGURATION
# ==========================================

window_sizes = np.array(window_sizes)

metrics = ["Precision", "Recall", "F1"]


# Colors for metrics
metric_colors = {
    "Precision": "#55A868",
    "Recall": "#8172B2",
    "F1": "#C44E52"
}

# Colors for feature sets
feature_colors = {
    "I": "#4C72B0",
    "II": "#DD8452",
    "III": "#55A868"
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})


# ==========================================
# GENERAL PLOTTING FUNCTION
# ==========================================

def plot_comparison(
    results,
    fixed_type,
    fixed_value,
    save=False
):

    fig, axes = plt.subplots(2, 1,
    figsize=(3.5, 5.5),
    sharex=True,
    sharey=True
)

    for ax, model in zip(axes, models):

        if fixed_type == "feature":

            # Fixed feature set, compare metrics
            for metric in metrics:

                values = results[model_names[model]][fixed_value][metric]

                ax.plot(
                    window_sizes,
                    values,
                    color=metric_colors[metric],
                    marker="o",
                    linestyle="-",
                    linewidth=1.5,
                    markersize=4,
                    label=metric
                )

        elif fixed_type == "metric":

            # Fixed metric, compare feature sets
            for feature in features:

                values = results[model_names[model]][feature][fixed_value]

                ax.plot(
                    window_sizes,
                    values,
                    color=feature_colors[feature],
                    marker="o",
                    linestyle="-",
                    linewidth=1.5,
                    markersize=4,
                    label=f"Feature set {feature}"
                )

        ax.set_title(model_names[model])
        ax.set_ylabel("Metric value")

        ax.set_xticks(window_sizes)
        ax.set_ylim(0, 1)

        ax.grid(
            True,
            alpha=0.3,
            linestyle="-"
        )

        #ax.legend(
        #    loc="lower right",
        #    frameon=True
        #)

    axes[-1].set_xlabel("Window length")

    #plt.tight_layout()
    
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        fontsize=8
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save:

        filename = f"{fixed_type}_{fixed_value}"

        #fig.savefig(f"{filename}.pdf",bbox_inches="tight")
        fig.savefig(f"{filename}.png",dpi=300,bbox_inches="tight")

    plt.show()


# ==========================================
# FIGURES 1–3: FIXED FEATURE SET
# ==========================================

for feature in features:

    plot_comparison(
        results,
        fixed_type="feature",
        fixed_value=feature,
        save=True
    )


# ==========================================
# FIGURES 4–6: FIXED METRIC
# ==========================================

for metric in metrics:

    plot_comparison(
        results,
        fixed_type="metric",
        fixed_value=metric,
        save=True
    )

