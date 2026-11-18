#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saraC
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from scipy import signal as sig

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#%%

# Nombres de carpetas
exp_name = "w80_s1_lstm0.3"
exp_dir = f"experiments/{exp_name}"
arrays_dir = f"{exp_dir}/arrays"
models_dir = f"{exp_dir}/models"
features_dir = f"{exp_dir}/features"

os.makedirs(features_dir, exist_ok=True)

with open(f"{exp_dir}/config.json", "r") as f:
    config = json.load(f)

#Parámetros de exp elegido
w_l = config["window_length"]
stride = config["stride"]
lstm_fraction = config["lstm_fraction"]

# Load files
data = np.load(f"{arrays_dir}/windows.npz")

X_train = data["X_train"]
X1_train = data["X1_train"]

X_test = data["X_test"]
X1_test = data["X1_test"]

info_train = pd.read_pickle(f"{arrays_dir}/info_train.pkl")
info_test = pd.read_pickle(f"{arrays_dir}/info_test.pkl")

# Crear dataset
dataset_train = info_train.copy()
dataset_test = info_test.copy()

dataset_train["split"] = "train"
dataset_test["split"] = "test"

dataset_train["target_value"] = X1_train
dataset_test["target_value"] = X1_test

#%%
# Features I

def diff_var(array):
    return np.var(np.diff(array))

def diff2_var(array):
    return np.var(np.diff(array, n=2))

transformations = {
    "feat_mean": np.mean,
    "feat_var": np.var,
    "feat_std": np.std,
    "feat_diff_var": diff_var,
    "feat_diff2_var": diff2_var,
}

for key, func in transformations.items():
    dataset_train[key] = np.apply_along_axis(func, axis=1, arr=X_train)
    dataset_test[key] = np.apply_along_axis(func, axis=1, arr=X_test)


#%%

## Features II

# SMA error
dataset_train["feat_SMA_err"] = X_train.mean(axis=1) - X1_train
dataset_test["feat_SMA_err"] = X_test.mean(axis=1) - X1_test

#LSTM error
dataset_train["feat_LSTM_err"] = np.nan
dataset_test["feat_LSTM_err"] = np.nan

for ch in tqdm(dataset_train["channel"].unique()):

    model_path = f"{models_dir}/{ch}.keras"

    if not os.path.exists(model_path):
        print(f"Model not found for {ch}")
        continue

    model = tf.keras.models.load_model(model_path)

    channel_mask_train = dataset_train["channel"] == ch
    channel_mask_test = dataset_test["channel"] == ch

    pred_train = model.predict(X_train[channel_mask_train.values][..., None], verbose=0).flatten()
    
    pred_test = model.predict(X_test[channel_mask_test.values][..., None], verbose=0).flatten()

    dataset_train.loc[channel_mask_train, "feat_LSTM_err"] = (pred_train - X1_train[channel_mask_train.values])

    dataset_test.loc[channel_mask_test, "feat_LSTM_err"] = (pred_test - X1_test[channel_mask_test.values])


# FFT std
hann = sig.windows.hann(w_l)

def ft_std_window(w):
    return np.std((2 / w_l) * (1 / w_l) * np.abs(np.fft.rfft((w - np.mean(w)) * hann)))

dataset_train["feat_FT_std"] = np.apply_along_axis(ft_std_window, axis=1, arr=X_train)

dataset_test["feat_FT_std"] = np.apply_along_axis(ft_std_window, axis=1, arr=X_test)

#%%
# Create df and save files
dataset = pd.concat([dataset_train, dataset_test], ignore_index=True)
dataset.to_csv(f"{features_dir}/dataset_features.csv", index=False)
dataset.to_pickle(f"{features_dir}/dataset_features.pkl")

print(dataset.shape)
print(dataset.columns)








