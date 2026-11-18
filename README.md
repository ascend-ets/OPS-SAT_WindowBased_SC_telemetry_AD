# OPS-SAT_WindowBased_SC_telemetry_AD
This repository contains the code used to adapt the OPS-SAT-AD telemetry data to a fixed-window anomaly detection workflow. The pipeline prepares the train and test windows, calculates statistical and prediction-based features, trains supervised classifiers, and generates window-level performance figures.

## Pipeline

Run the scripts in this order:

1. `datasplit_LSTM.py` — normalizes telemetry values by channel, selects nominal training segments for the LSTM, creates windows without crossing segment boundaries, and trains one LSTM model per channel.
2. `features_calculation.py` — calculates the feature sets for each train and test window, including the LSTM prediction error.
3. `AD_detector.py` — trains Random Forest and AdaBoost classifiers and saves their test predictions.
4. `metrics.py` — reads the predictions across the configured window lengths and saves comparison figures.

## Repository layout

```text
.
├── data/
│   └── segments.csv
├── datasplit_LSTM.py
├── features_calculation.py
├── AD_detector.py
├── metrics.py
└── experiments/
    └── <experiment_name>/
        ├── config.json
        ├── arrays/
        ├── models/
        ├── plots/
        ├── features/
        └── classifier/
```

The data file is not included in this repository. Place `segments.csv` in `data/` before running the pipeline. The scripts use the columns `channel`, `segment`, `sampling`, `value`, `train`, `anomaly`, and `label`.

## Requirements

Use a Python version supported by the installed TensorFlow release. Install the packages used by the scripts in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy scikit-learn tensorflow tqdm pyod
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of the `source` command.

## Configure and run

Run the commands from the repository root so the relative paths to `data/` and `experiments/` resolve.

### 1. Create windows and train the LSTM models

In `datasplit_LSTM.py`, set the desired values for `sampling_LSTM`, `w_l`, `stride`, and `lstm_fraction`. Then run:

```bash
python datasplit_LSTM.py
```

The experiment folder name is generated as:

```text
w<w_l>_s<stride>_lstm<lstm_fraction>
```

For example, `w80_s1_lstm0.3`.

### 2. Calculate window features

Set `exp_name` in `features_calculation.py` to the experiment folder created in the previous step, then run:

```bash
python features_calculation.py
```

### 3. Train the classifiers

Set `exp_name` in `AD_detector.py` to the same experiment folder. Set `sel_features` to the feature set to evaluate:

- `I`: statistical features
- `II`: SMA error, LSTM prediction error, and Fourier magnitude feature
- `III`: combination of feature sets I and II

Run the script once for each feature set of interest:

```bash
python AD_detector.py
```

Each run saves the classifier models and test predictions under `experiments/<experiment_name>/classifier/<feature_set>/`.

### 4. Generate the figures

After running the pipeline for every window length listed in `metrics.py`, run:

```bash
python metrics.py
```

The script saves PNG figures in the current working directory. Update `window_sizes` and the experiment path pattern in `metrics.py` if the experiment settings differ from its defaults.

## Saved data
- `config.json` with the window and LSTM split settings.
- `arrays/windows.npz` contains the train and test windows and their next-sample targets.
- `arrays/info_train.csv` and `arrays/info_test.csv` contain the corresponding window metadata and labels.
- `models/` contains per-channel Keras LSTM models and training-loss figures.
- `features/dataset_features.csv` contains the calculated features for train and test windows.
- `classifier/feature-set`contains the classifier configuration and individual test prediction files.
- Window-level metric comparison figures from `metrics.py`.

## Reproducibility

The scripts define the experiment parameters and random seed in their source. Keep the experiment name and settings consistent across all four stages. The metrics script requires predictions for every configured window length, classifier, and feature set.
