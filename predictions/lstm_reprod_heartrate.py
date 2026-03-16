"""
Created on Tue Sep  7 21:30:05 2021

@author: harisushehu

Slight modifications of the original code were needed to run the code on the current data.
"""

from __future__ import annotations

import csv
import glob
import itertools
import os
import re
from csv import writer
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.python.framework import ops

ops.reset_default_graph()

# -------------------------------------------------------
# Create timestamp output directory
# -------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = os.path.join("predictions/output/lstm_reprod_hr", f"run_{timestamp}")
VAL_PLOT_DIR = os.path.join(OUTPUT_DIR, "val_patient_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_PLOT_DIR, exist_ok=True)

print("Results will be saved to:", OUTPUT_DIR)

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

TRAIN_PATH = "data/train"
VAL_PATH = "data/val"

# -------------------------------------------------------
# TensorFlow session config
# -------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(
    config=tf.compat.v1.ConfigProto(log_device_placement=True)
)
tf.compat.v1.keras.backend.set_session(sess)


@dataclass
class PreparedValRows:
    """Container for prepared validation rows and metadata.

    Attributes:
        X: Scaled feature matrix in 2D shape.
        y_true_original: Ground-truth heart rate in original scale.
        meta: Row-level metadata aligned with X and y.
    """

    X: np.ndarray
    y_true_original: np.ndarray
    meta: pd.DataFrame


def nrmse(rmse: float, y_test: np.ndarray) -> float:
    """Compute normalized root mean squared error.

    Args:
        rmse: Root mean squared error.
        y_test: Ground-truth target values.

    Returns:
        The normalized RMSE value.
    """
    nrmse_value = rmse / (max(y_test) - min(y_test))
    return nrmse_value[0]


def model(X: np.ndarray) -> Sequential:
    """Build and compile the LSTM model.

    Args:
        X: Training input array used to infer the input shape.

    Returns:
        A compiled Keras Sequential model.
    """
    regressor = Sequential()

    # Add first layer
    regressor.add(
        LSTM(units=150, return_sequences=True, input_shape=(X.shape[1], 1))
    )
    regressor.add(Dropout(0.2))

    # Add second layer
    regressor.add(LSTM(units=150, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Add third layer
    regressor.add(Dense(units=150))
    regressor.add(Dropout(0.2))

    # Add fourth layer
    regressor.add(LSTM(units=150))
    regressor.add(Dropout(0.2))

    # Add output layer
    regressor.add(Dense(units=1))
    regressor.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )

    return regressor


def append_list_as_row(file_name: str, list_of_elem: List[str]) -> None:
    """Append one row to a CSV file.

    Args:
        file_name: Path to the CSV file.
        list_of_elem: Row contents to append.
    """
    with open(file_name, "a+", newline="") as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


def parse_participant_id(filename: str) -> Optional[str]:
    """Extract participant ID from a CSV filename.

    Args:
        filename: Input CSV filename.

    Returns:
        The participant ID if found, otherwise None.
    """
    basename = os.path.basename(filename)
    match = re.search(r"P(\d{3})", basename)

    if match is None:
        return None

    return match.group(1)


def parse_trial_id(filename: str) -> Optional[int]:
    """Extract trial ID from a CSV filename.

    Args:
        filename: Input CSV filename.

    Returns:
        The trial ID if found, otherwise None.
    """
    basename = os.path.basename(filename)
    match = re.search(r"T(\d+)", basename)

    if match is None:
        return None

    return int(match.group(1))


def prepare_external_val_rows(
    val_files: List[str],
    scaler_X: StandardScaler,
    target_column: str = "heartrate_mean",
) -> PreparedValRows:
    """Prepare all rows from the external validation set.

    The function keeps the target in original scale for plotting and applies the
    train-fitted feature scaler to the validation features.

    Args:
        val_files: List of validation CSV file paths.
        scaler_X: Feature scaler fitted on the current training fold.
        target_column: Name of the target column.

    Returns:
        Prepared validation rows with scaled features, original target values,
        and row-level metadata.

    Raises:
        ValueError: If no usable validation rows are found.
    """
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    meta_parts: List[pd.DataFrame] = []

    for filepath in val_files:
        df = pd.read_csv(filepath, encoding="ISO-8859-1", header=0).fillna(0)

        if target_column not in df.columns:
            continue

        X_df = df.loc[:, df.columns != target_column].copy()
        y_df = df.loc[:, df.columns == target_column].copy()

        # Keep only valid target rows
        valid_mask = np.isfinite(y_df[target_column].to_numpy())
        X_df = X_df.loc[valid_mask].reset_index(drop=True)
        y_df = y_df.loc[valid_mask].reset_index(drop=True)

        if len(X_df) == 0:
            continue

        X_scaled = scaler_X.transform(X_df)
        y_original = y_df[target_column].to_numpy(dtype=np.float32)

        participant_id = parse_participant_id(filepath) or "unknown"
        trial_id = parse_trial_id(filepath)
        if trial_id is None:
            trial_id = -1

        meta_df = pd.DataFrame(
            {
                "patient_id": [participant_id] * len(X_df),
                "trial_id": [trial_id] * len(X_df),
                "filepath": [filepath] * len(X_df),
                "row_in_trial": np.arange(len(X_df), dtype=int),
            }
        )

        x_parts.append(X_scaled.astype(np.float32))
        y_parts.append(y_original)
        meta_parts.append(meta_df)

    if not x_parts:
        raise ValueError("No usable validation rows were found in data/val.")

    X = np.vstack(x_parts)
    y_true_original = np.concatenate(y_parts)
    meta = pd.concat(meta_parts, axis=0, ignore_index=True)

    return PreparedValRows(X=X, y_true_original=y_true_original, meta=meta)


def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    """Reshape a 2D input matrix for LSTM usage.

    Args:
        X: Input feature matrix of shape (n_samples, n_features).

    Returns:
        Reshaped matrix of shape (n_samples, n_features, 1).
    """
    return np.reshape(X, [X.shape[0], X.shape[1], 1])


def inverse_transform_targets(
    y_scaled: np.ndarray,
    scaler_y: StandardScaler,
) -> np.ndarray:
    """Transform scaled target values back to original scale.

    Args:
        y_scaled: Target values in scaled space.
        scaler_y: Target scaler fitted on the training fold.

    Returns:
        Target values in original scale.
    """
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    return scaler_y.inverse_transform(y_scaled).reshape(-1)


def save_patient_validation_plots(
    pred_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save one plot per validation patient across all trials.

    Each plot concatenates all available trials of one patient in ascending
    trial order. Trial boundaries are marked by dashed vertical lines.

    Args:
        pred_df: DataFrame containing metadata, y_true, and y_pred.
        output_dir: Directory where plots should be stored.
    """
    os.makedirs(output_dir, exist_ok=True)

    for patient_id, patient_df in pred_df.groupby("patient_id", sort=True):
        patient_df = patient_df.sort_values(
            ["trial_id", "row_in_trial"]
        ).reset_index(drop=True)

        y_true = patient_df["y_true"].to_numpy()
        y_pred = patient_df["y_pred"].to_numpy()

        # Store the cumulative end positions of trials to draw separators
        boundary_positions: List[int] = []
        cumulative = 0

        for _, trial_df in patient_df.groupby("trial_id", sort=True):
            cumulative += len(trial_df)
            boundary_positions.append(cumulative)

        plt.figure(figsize=(18, 6))
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Predicted")

        # Draw dashed lines between trials, but not after the last one
        for boundary in boundary_positions[:-1]:
            plt.axvline(boundary, linestyle="--", linewidth=1.0, color="gray")

        plt.title(f"Validation - Patient {patient_id} - Heart Rate")
        plt.xlabel("Concatenated time steps across trials")
        plt.ylabel("Heart Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"patient_{patient_id}_heart_rate.png"),
            dpi=150,
        )
        plt.close()


# -------------------------------------------------------
# CSV result file
# -------------------------------------------------------

csvFileName = f"{OUTPUT_DIR}/heart_rate_results.csv"

if not os.path.exists(csvFileName):
    with open(csvFileName, "w", newline="") as f:
        header = ["Fold", "RMSE", "NRMSE"]
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

print("Reading data...")

path = TRAIN_PATH

if path == "dataset path":
    print("Please insert the database path to continue...")

else:
    all_files = glob.glob(path + "/*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, encoding="ISO-8859-1", header=0)
        li.append(df)

    dataset = pd.concat(li, axis=0, ignore_index=True)

    # Replace NaN with 0
    dataset = dataset.fillna(0)

    print("Evaluating...")

    X = dataset.iloc[:, df.columns != "heartrate_mean"]
    y = dataset.iloc[:, df.columns == "heartrate_mean"].values

    print("X is :", X.shape)
    print("y is :", y.shape)

    print("Reading data for preprocessing...")
    all_files = sorted(glob.glob(path + "/*.csv"))

    # Files per participant based on the files that actually exist
    participant_to_files = {}

    for filename in all_files:
        basename = os.path.basename(filename)

        # Expects e.g. P001, P023, ...
        match = re.search(r"P(\d{3})", basename)

        if match is None:
            continue

        participant_id = match.group(1)

        if participant_id not in participant_to_files:
            participant_to_files[participant_id] = []

        reader = pd.read_csv(filename, encoding="ISO-8859-1", header=0)
        participant_to_files[participant_id].append(reader)

    participant_ids = sorted(participant_to_files.keys())

    print("Number of participants with data:", len(participant_ids))

    # Distribute participant groups across available folds
    participant_groups = [
        participant_to_files[participant_id] for participant_id in participant_ids
    ]

    # At most 5 folds, but never more folds than participants
    n_folds = min(5, len(participant_groups))

    split_groups = [participant_groups[i::n_folds] for i in range(n_folds)]

    # Flatten every fold group back into one list of DataFrames
    list_of_lists = []
    for fold_idx, fold in enumerate(split_groups):
        flat_fold = []
        for participant_file_list in fold:
            flat_fold.extend(participant_file_list)
        list_of_lists.append(flat_fold)
        print(f"Fold {fold_idx}: {len(flat_fold)} files")

    # Read external validation files once
    external_val_files = sorted(glob.glob(os.path.join(VAL_PATH, "*.csv")))

    for i in range(len(list_of_lists)):
        print("***********Splitting test and train...")
        train_lists = list_of_lists[0:i] + list_of_lists[i + 1 :]
        print(len(train_lists))
        test_list = list_of_lists[i]
        print(len(test_list))

        print("***********Splitting train and eval...")
        second_list_of_lists = train_lists
        for j in range(0, 1):
            if i < len(list_of_lists) - 1:
                train_lists = second_list_of_lists[0:j] + second_list_of_lists[j + 1 :]
                val_list = second_list_of_lists[i]
            else:
                train_lists = second_list_of_lists[0:j] + second_list_of_lists[j + 1 :]
                val_list = second_list_of_lists[j]

        train = list(itertools.chain(*train_lists))

        train_data = pd.concat(train, axis=0, ignore_index=True)

        # Replace NaN with 0
        train_data = train_data.fillna(0)

        X_train = train_data.iloc[:, df.columns != "heartrate_mean"]
        y_train = train_data.iloc[:, df.columns == "heartrate_mean"].values

        # Scale train
        scaler_X_train = StandardScaler()
        scaler_y_train = StandardScaler()

        # Normalize train X and y
        X_train = scaler_X_train.fit_transform(X_train)
        y_train = scaler_y_train.fit_transform(y_train)

        test_data = pd.concat(test_list, axis=0, ignore_index=True)

        # Replace NaN with 0
        test_data = test_data.fillna(0)

        X_test = test_data.iloc[:, df.columns != "heartrate_mean"]
        y_test = test_data.iloc[:, df.columns == "heartrate_mean"].values

        # Keep original logic for test scaling
        scaler_X_test = StandardScaler()
        scaler_y_test = StandardScaler()

        X_test = scaler_X_test.fit_transform(X_test)
        y_test = scaler_y_test.fit_transform(y_test)

        val_data = pd.concat(val_list, axis=0, ignore_index=True)

        # Replace NaN with 0
        val_data = val_data.fillna(0)

        X_val = val_data.iloc[:, df.columns != "heartrate_mean"]
        y_val = val_data.iloc[:, df.columns == "heartrate_mean"].values

        # Keep original logic for inner validation scaling
        scaler_X_val = StandardScaler()
        scaler_y_val = StandardScaler()

        X_val = scaler_X_val.fit_transform(X_val)
        y_val = scaler_y_val.fit_transform(y_val)

        X_train = np.array(X_train)
        X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1], 1])

        X_val = np.array(X_val)
        X_val = np.reshape(X_val, [X_val.shape[0], X_val.shape[1], 1])

        print("Shapes of X_train, X_test, and X_val...")
        print("X_train :", X_train.shape)
        print("X_test :", X_test.shape)
        print("X_val :", X_val.shape)

        # Training
        regressor = model(X_train)
        regressor.fit(
            X_train,
            y_train,
            epochs=1,
            verbose=1,
            batch_size=64,
            validation_data=(X_val, y_val),
        )

        # -------------------------------------------------------
        # Original fold-test prediction + metrics remain unchanged
        # -------------------------------------------------------
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)

        print("Results on Fold: " + str(i))
        print("Root mean squared error", rmse)

        nrmse_val = nrmse(rmse, y_test)
        print("Normalized root mean squared error", nrmse_val)

        row_contents = [str(i), str(rmse), str(nrmse_val)]
        append_list_as_row(csvFileName, row_contents)

        # -------------------------------------------------------
        # New external validation plots on actual heart rate scale
        # -------------------------------------------------------
        print(f"Creating external validation patient plots for fold {i}...")

        val_rows = prepare_external_val_rows(
            val_files=external_val_files,
            scaler_X=scaler_X_train,
            target_column="heartrate_mean",
        )

        X_val_external = reshape_for_lstm(val_rows.X)
        y_pred_val_scaled = regressor.predict(X_val_external, verbose=0)

        # Transform predictions back to actual heart rate scale
        y_pred_val_original = inverse_transform_targets(
            y_pred_val_scaled,
            scaler_y_train,
        )

        pred_df = val_rows.meta.copy()
        pred_df["y_true"] = val_rows.y_true_original
        pred_df["y_pred"] = y_pred_val_original

        fold_plot_dir = os.path.join(VAL_PLOT_DIR, f"fold_{i}")
        save_patient_validation_plots(
            pred_df=pred_df,
            output_dir=fold_plot_dir,
        )

    print(f"Done. Results saved to: {OUTPUT_DIR}")