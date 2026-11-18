#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import History, EarlyStopping, Callback
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Input
import os
import json
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#%%
# CHARACTERISTICS CHOSEN FOR THE EXTRACTION AND LSTM TRAINING
df = pd.read_csv('data/segments.csv')
# Choice of sampling ('1', '5', or 'both')
sampling_LSTM = 'both'

# Window length and predicted values
w_l = 60
stride = 1
n_p = 1
lstm_fraction=0.3

exp_name = f"w{w_l}_s{stride}_lstm{lstm_fraction}"
exp_dir = f"experiments/{exp_name}"

arrays_dir = f"{exp_dir}/arrays"
models_dir = f"{exp_dir}/models"
plots_dir = f"{exp_dir}/plots"

random_seed = 42

#%%
# ADDING INTERESTING COLUMNS

# choosing the sampling
if sampling_LSTM == '1':
    df = df[df['sampling'] == 1]
elif sampling_LSTM == '5':
    df = df[df['sampling'] == 5]

# adding channel_type column for future calculations
df['channel_type'] = ''
for i in df.channel.unique():
    if i in df.channel.sort_values().unique()[:3]:
        df.loc[df['channel'] == i, 'channel_type'] = 'magnetometer'
    else:
        df.loc[df['channel'] == i, 'channel_type'] = 'photodiode'

# adding the normalized values' columns

# Normalize each channel using nominal training samples only

df['normalized_value'] = np.nan
scalers = {}

for channel in df['channel'].unique():

    channel_mask = df['channel'] == channel

    train_nominal_mask = (channel_mask & (df['train'] == 1) & (df['anomaly'] == 0))

    channel_type = df.loc[channel_mask, 'channel_type'].iloc[0]

    if channel_type == 'magnetometer':
        scaler = MaxAbsScaler()
    else:
        scaler = MinMaxScaler()

    # Fit only on nominal training samples
    train_values = df.loc[train_nominal_mask, 'value'].to_numpy().reshape(-1, 1)

    if len(train_values) == 0:
        raise ValueError(f"No nominal training samples for {channel}")

    scaler.fit(train_values)

    all_values = df.loc[channel_mask, 'value'].to_numpy().reshape(-1, 1)
    df.loc[channel_mask, 'normalized_value'] = scaler.transform(all_values).ravel()
    scalers[channel] = scaler

# remove channels
df = df[~df.channel.isin(['CADC0886','CADC0890','CADC0884'])]

#%%
############# some stats #############

def segment_window_metrics(df, w_l):
    
    # Resumen por canal/segmento
    seg = (df.groupby(['channel', 'segment']).agg(split=('train', 'first'), n_samples=('segment', 'size'), n_anom=('label', 'sum')).reset_index())
    seg['split'] = np.where(seg['split'] == 1, 'train', 'test')
    seg['type'] = np.where(seg['n_anom'] == 0, 'nominal', 'anomalous')

    # Ventanas posibles por segmento
    seg['n_windows'] = np.maximum(seg['n_samples'] - w_l + 1, 0)

    # Métricas finales
    summary = (seg.groupby(['channel', 'split', 'type']).agg(
            n_segments=('segment', 'count'),
            total_samples=('n_samples', 'sum'),
            total_windows=('n_windows', 'sum'),
            min_samples=('n_samples', 'min'),
            max_samples=('n_samples', 'max'),
            max_windows=('n_windows', 'max')
        ).reset_index().sort_values(['channel', 'split', 'type']))

    return summary, seg

summary, segment_detail = segment_window_metrics(df, w_l=w_l)

#%% Split segments for LSTM

def split_by_channel_segment_lstm(df, lstm_fraction, random_state):
    
    lstm_segments_all = []

    for channel in df['channel'].unique():

        segment_anomaly = (df[(df['channel'] == channel) & 
                              (df['train'] == 1)].groupby('segment')['anomaly'].sum())

        nominal_train_segments = segment_anomaly[segment_anomaly == 0].index.values

        if len(nominal_train_segments) == 0:
            continue
        
        # Cantidad de segmentos para la LSTM % de nominal train 
        n_select = max(1, int(np.floor(len(nominal_train_segments) * lstm_fraction)))
        #Selecciona n_select segmentos aleatorios (default_rng para reproducibilidad)
        selected_lstm_segments = np.random.default_rng(random_state).choice(
            nominal_train_segments,
            size = n_select,
            replace = False)
        # Lista de segmentos para LSTM: [(canal, #segmento)]
        lstm_segments_all.extend([(channel, seg) for seg in selected_lstm_segments])
            
    selected = set(lstm_segments_all)
    
    return selected

selected = split_by_channel_segment_lstm(df, lstm_fraction=lstm_fraction, random_state=random_seed)


#%%
# Split selected LSTM segments into training and validation

lstm_train_segments = set()
lstm_val_segments = set()

rng = np.random.default_rng(random_seed)

for channel in df['channel'].unique():

    channel_segments = [seg for ch, seg in selected if ch == channel]

    if len(channel_segments) < 2:
        raise ValueError(f"{channel}: insufficient segments for independent LSTM training and validation.")

    train_seg, val_seg = train_test_split(channel_segments, test_size=0.33, random_state=random_seed)
    lstm_train_segments.update((channel, seg) for seg in train_seg)
    lstm_val_segments.update((channel, seg) for seg in val_seg)

#%% Mae all possible windows

def make_windows(df, window_length, stride):

    X = []
    y = []
    window_info = []
    excluded_segments = []

    for (channel, segment), group in df.groupby(['channel', 'segment'], sort=False):

        # mantener el orden temporal original
        group = group.sort_index()
        n = len(group)

        # necesito al menos window_length + 1 muestras
        if n <= window_length:
            excluded_segments.append({
                'channel': channel,
                'segment': segment,
                'n_samples': n,
                'train': group['train'].iloc[0]
            })
            continue

        values = group['normalized_value'].values

        for start in range(0, n - window_length, stride):

            end = start + window_length

            # ventana
            X.append(values[start:end])

            # siguiente valor
            y.append(values[end])

            window_info.append({
                'channel': channel,
                'segment': segment,
                'start_idx': group.index[start],
                'end_idx': group.index[end - 1],
                'target_idx': group.index[end],
                'train': group['train'].iloc[0],
                'window_label_sum': group['anomaly'].iloc[start:end].sum(),
                'window_label': int(
                    group['anomaly'].iloc[start:end].sum() > 0
                ),
                'target_anomaly': group['anomaly'].iloc[end]
            })

    return (
        np.asarray(X),
        np.asarray(y),
        pd.DataFrame(window_info),
        pd.DataFrame(excluded_segments)
    )


X, X_1, window_info, excluded_segments = make_windows(df,window_length=w_l,stride=stride)
#%%

# Identify the segment corresponding to each window

window_segments = list(zip(window_info['channel'], window_info['segment']))
lstm_train_mask = np.array([seg in lstm_train_segments for seg in window_segments])
lstm_val_mask = np.array([seg in lstm_val_segments for seg in window_segments])
lstm_mask = lstm_train_mask | lstm_val_mask

train_mask = ((window_info['train'] == 1).values & ~lstm_mask)
test_mask = (window_info['train'] == 0).values
   
# LSTM training

X_lstm_train = X[lstm_train_mask][..., None]
X1_lstm_train = X_1[lstm_train_mask]
info_lstm_train = window_info.loc[lstm_train_mask].reset_index(drop=True)

# LSTM validation
X_lstm_val = X[lstm_val_mask][..., None]
X1_lstm_val = X_1[lstm_val_mask]
info_lstm_val = window_info.loc[lstm_val_mask].reset_index(drop=True)


# Remaining training and test data

X_train = X[train_mask]
X1_train = X_1[train_mask]

X_test = X[test_mask]
X1_test = X_1[test_mask]

info_train = window_info.loc[train_mask].reset_index(drop=True)

info_test = window_info.loc[test_mask].reset_index(drop=True)

print("X_lstm:", X_lstm_train.shape)
print("X1_lstm:", X1_lstm_train.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("LSTM channels:", info_lstm_train['channel'].nunique())
print("Train channels:", info_train['channel'].nunique())
print("Test channels:", info_test['channel'].nunique())

#%%

#  Guardar arreglos y experimento
os.makedirs(arrays_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

config = {
    "sampling_LSTM": sampling_LSTM,
    "window_length": w_l,
    "stride": stride,
    "n_p": n_p,
    "lstm_fraction": lstm_fraction,
    "random_state": random_seed
}

with open(f"{exp_dir}/config.json", "w") as f:
    json.dump(config, f, indent=4)

np.savez_compressed(f"{arrays_dir}/windows.npz",X_train=X_train, X1_train=X1_train, X_test=X_test, X1_test=X1_test,)

info_train.to_csv(f"{arrays_dir}/info_train.csv", index=False)
info_test.to_csv(f"{arrays_dir}/info_test.csv", index=False)
#%%
for ch in info_lstm_train['channel'].unique():

    n_train_segments = info_lstm_train.loc[info_lstm_train['channel'] == ch,'segment' ].nunique()
    n_val_segments = info_lstm_val.loc[info_lstm_val['channel'] == ch,'segment'].nunique()
    n_train_windows = (info_lstm_train['channel'] == ch).sum()
    n_val_windows = (info_lstm_val['channel'] == ch).sum()

    print(
        f"{ch}: "
        f"Train segments={n_train_segments}, "
        f"Val segments={n_val_segments}, "
        f"Train windows={n_train_windows}, "
        f"Val windows={n_val_windows}"
    )


#%%
# LSTM TRAINING

model_dir = models_dir
os.makedirs(model_dir, exist_ok=True)

models = {}
histories = {}

for ch in info_lstm_train['channel'].unique():

    train_mask_ch = (info_lstm_train['channel'] == ch).values
    val_mask_ch = (info_lstm_val['channel'] == ch).values

    x_train = X_lstm_train[train_mask_ch]
    y_train = X1_lstm_train[train_mask_ch]

    x_val = X_lstm_val[val_mask_ch]
    y_val = X1_lstm_val[val_mask_ch]

    if len(x_train) == 0 or len(x_val) == 0:
        print(f"Skipping {ch}: insufficient data")
        continue

    cbs = [EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0003,restore_best_weights=True,verbose=0)]

    model = Sequential([
        Input(shape=(x_train.shape[1], x_train.shape[2])),
        LSTM(16),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mse']
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=35,
        callbacks=cbs,
        verbose=1
    )

    model.save(f"{models_dir}/{ch}.keras")

    models[ch] = model
    histories[ch] = history

    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{ch} train vs validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(f"{plots_dir}/{ch}_loss.png", dpi=300, bbox_inches="tight")
    plt.show()


