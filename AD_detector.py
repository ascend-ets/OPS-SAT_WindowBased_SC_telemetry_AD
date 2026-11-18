#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saraC
"""

import os
import random
import joblib
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

#%% Select experiment and load raw data and windows

exp_name = "w80_s1_lstm0.3"
exp_dir = f"experiments/{exp_name}"
arrays_dir = f"{exp_dir}/arrays"
features_dir = f"{exp_dir}/features"

data = np.load(f"{arrays_dir}/windows.npz")
X_train = data["X_train"]
X_test = data["X_test"]

info_train = pd.read_pickle(f"{arrays_dir}/info_train.pkl")
info_test = pd.read_pickle(f"{arrays_dir}/info_test.pkl")

y_train = info_train['window_label']
y_test = info_test['window_label']

df = pd.read_csv(f"{features_dir}/dataset_features.csv")
features_col = df.filter(like="feat").columns.tolist()

#%% Define the set of features used to train classifer and create train and test set

sel_features = "III" #"windows" #"all" # "subset",# "windows"

# Masks
train_mask = df["train"] == 1
test_mask  = df["train"] == 0


if sel_features == "I":
    features = ['feat_mean', 'feat_var', 'feat_std', 'feat_diff_var', 'feat_diff2_var']
elif sel_features == "II":
    features = ['feat_SMA_err', 'feat_LSTM_err', 'feat_FT_std']
elif sel_features == "III":
    features = ['feat_mean', 'feat_var', 'feat_std', 'feat_diff_var', 
                'feat_diff2_var', 'feat_SMA_err', 'feat_LSTM_err', 'feat_FT_std']
else:
    raise ValueError(f"Unknown sel_features = {sel_features}")
            
    
if sel_features == "windows":
    X_tr = X_train
    X_te = X_test
    classifier_config = {
        "feature_mode": "windows",
        "features": "raw_window_values",
        "window_length": X_tr.shape[1],
        "n_features": X_tr.shape[1]
    }
else:
    
    X_tr = df.loc[train_mask, features].values
    X_te = df.loc[test_mask, features].values
    
    classifier_config = {
        "feature_mode": sel_features,
        "features": features,
        "n_features": len(features)
    }

#%% Create the classffication summary

classifier_dir = f"{exp_dir}/classifier/{sel_features}"
os.makedirs(classifier_dir, exist_ok=True)

with open(f"{classifier_dir}/classifier_config.json", "w") as f:
    json.dump(classifier_config, f, indent=4)
#%% Define the classification models

# Libraries
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Models and parameters
models = {
    "RF": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "ADA": AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    )
}

#%%


predictions_all = []

for model_name, model in models.items():

    print("################################")
    print(f"Training {model_name}")

    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_te)
    else:
        y_score = y_pred

    # guardar modelo
    joblib.dump(
        model,
        f"{classifier_dir}/{model_name}.pkl"
    )

    # guardar predicciones
    pred_df = info_test.copy()
    pred_df["model"] = model_name
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_pred
    pred_df["y_score"] = y_score

    pred_df.to_csv(
        f"{classifier_dir}/predictions_{model_name}.csv",
        index=False
    )

    predictions_all.append(pred_df)

predictions_all = pd.concat(predictions_all, ignore_index=True)

predictions_all.to_csv(
    f"{classifier_dir}/predictions_all.csv",
    index=False
)
