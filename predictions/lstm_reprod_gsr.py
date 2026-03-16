"""
Created on Tue Sep  7 21:30:05 2021

@author: harisushehu

Slight modifications of the original code were needed to run the code on the current data.
"""

import csv
import glob
import math
import os
import re
from csv import writer
from math import sqrt
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.python.framework import ops

ops.reset_default_graph()

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# =============================================================================
# Hardcoded paths
# =============================================================================
TRAIN_PATH = "data/train"
VAL_PATH = "data/val"
OUTPUT_DIR = "predictions/output/lstm_reprod_gsr"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "rmse_nrmse_results.csv")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "prediction_plots")

# =============================================================================
# TensorFlow / GPU config (kept close to original script)
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(
    config=tf.compat.v1.ConfigProto(log_device_placement=True)
)
tf.compat.v1.keras.backend.set_session(sess)


def nrmse(rmse: float, y_test: np.ndarray) -> float:
    """Compute normalized RMSE based on the value range of y_test.

    Args:
        rmse: Root mean squared error.
        y_test: Ground-truth target values with shape (n_samples, 1).

    Returns:
        The normalized RMSE as a float.
    """
    value_range = max(y_test) - min(y_test)
    return (rmse / value_range)[0]


def model(X: np.ndarray) -> Sequential:
    """Create the LSTM model.

    Args:
        X: Input array used only to infer the input shape.

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
        list_of_elem: Row values to append.
    """
    with open(file_name, "a+", newline="", encoding="utf-8") as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


def ensure_output_structure() -> None:
    """Create output folders and initialize the results CSV if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            header = ["Fold", "RMSE", "NRMSE"]
            filewriter = csv.DictWriter(f, fieldnames=header)
            filewriter.writeheader()


def extract_participant_id(file_path: str) -> Optional[str]:
    """Extract a 3-digit participant ID from a filename.

    The function tries a few common filename patterns such as:
    - P001
    - _001
    - 001

    Args:
        file_path: Full file path.

    Returns:
        The extracted participant ID as a zero-padded 3-digit string,
        or None if no ID could be found.
    """
    filename = os.path.basename(file_path)

    # Preferred pattern: P001 / p001
    match = re.search(r"[Pp](\d{3})", filename)
    if match:
        return match.group(1)

    # Fallback: any standalone 3-digit number
    match = re.search(r"(?<!\d)(\d{3})(?!\d)", filename)
    if match:
        return match.group(1)

    return None


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file with the original encoding settings.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(path, encoding="ISO-8859-1", header=0)


def get_participant_file_map(folder_path: str) -> Dict[str, List[str]]:
    """Group CSV files by participant ID.

    Args:
        folder_path: Folder containing participant CSV files.

    Returns:
        A dictionary mapping participant IDs to sorted file lists.
    """
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    participant_to_files: Dict[str, List[str]] = {}

    for file_path in all_files:
        participant_id = extract_participant_id(file_path)
        if participant_id is None:
            print(f"Warning: Could not extract participant ID from {file_path}")
            continue

        participant_to_files.setdefault(participant_id, []).append(file_path)

    return participant_to_files


def build_fold_lists(
    participant_to_files: Dict[str, List[str]], n_splits: int = 5
) -> List[List[pd.DataFrame]]:
    """Build fold lists dynamically based on available participants.

    Each fold contains the CSV DataFrames of a subset of participants.
    This replaces the original fixed '29 participants per split' logic.

    Args:
        participant_to_files: Mapping from participant ID to file paths.
        n_splits: Number of folds to create.

    Returns:
        A list of folds. Each fold is a list of DataFrames.
    """
    participant_ids = sorted(participant_to_files.keys())
    participant_chunks = np.array_split(participant_ids, n_splits)

    list_of_lists: List[List[pd.DataFrame]] = []

    for chunk in participant_chunks:
        fold_dataframes: List[pd.DataFrame] = []

        for participant_id in chunk:
            for file_path in participant_to_files[participant_id]:
                reader = load_csv(file_path)
                fold_dataframes.append(reader)

        list_of_lists.append(fold_dataframes)

    return list_of_lists


def dataframe_to_xy(
    data: pd.DataFrame, target_column: str = "GSR_mean"
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a DataFrame into features and target arrays.

    Args:
        data: Input DataFrame.
        target_column: Name of the target column.

    Returns:
        Tuple of (X, y).
    """
    X = data.loc[:, data.columns != target_column]
    y = data.loc[:, data.columns == target_column].values
    return X.values, y


def scale_xy_separately(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale X and y separately with StandardScaler.

    This intentionally keeps the original script behavior:
    each split is scaled independently.

    Args:
        X: Feature matrix.
        y: Target array.

    Returns:
        Scaled (X, y).
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled


def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    """Reshape features for LSTM input.

    Args:
        X: 2D feature matrix of shape (n_samples, n_features).

    Returns:
        3D feature tensor of shape (n_samples, n_features, 1).
    """
    X = np.array(X)
    return np.reshape(X, [X.shape[0], X.shape[1], 1])


def prepare_dataframe_list(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of DataFrames and fill NaNs with zero.

    Args:
        dataframes: List of pandas DataFrames.

    Returns:
        Prepared concatenated DataFrame.
    """
    data = pd.concat(dataframes, axis=0, ignore_index=True)
    data = data.fillna(0)
    return data


def predict_val_patient_all_trials(
    regressor: Sequential,
    patient_id: str,
    patient_files: List[str],
    output_path: str,
    target_column: str = "GSR_mean",
) -> None:
    """Create one concatenated prediction plot for all trials of one patient.

    The trials are concatenated in order and separated visually by dashed
    vertical lines.

    Important:
    To stay close to the original script logic, scaling is done on the full
    concatenated patient validation data itself.

    Args:
        regressor: Trained Keras model.
        patient_id: Participant ID.
        patient_files: List of CSV files for that patient.
        output_path: Output path for the saved plot.
        target_column: Name of the target column.
    """
    trial_dfs: List[pd.DataFrame] = []
    trial_lengths: List[int] = []

    # Read and store all trials for this patient
    for file_path in sorted(patient_files):
        df_trial = load_csv(file_path).fillna(0)
        trial_dfs.append(df_trial)
        trial_lengths.append(len(df_trial))

    # Concatenate all trials
    patient_df = pd.concat(trial_dfs, axis=0, ignore_index=True)

    # Split into X / y
    X_patient, y_patient = dataframe_to_xy(patient_df, target_column=target_column)

    # Keep original-style split-wise scaling logic
    X_patient_scaled, y_patient_scaled = scale_xy_separately(X_patient, y_patient)

    # Reshape for LSTM prediction
    X_patient_scaled = reshape_for_lstm(X_patient_scaled)

    # Predict
    y_pred = regressor.predict(X_patient_scaled, verbose=0)

    # Flatten for plotting
    y_true_plot = y_patient_scaled.flatten()
    y_pred_plot = y_pred.flatten()

    # Create plot
    plt.figure(figsize=(16, 6))
    plt.plot(y_true_plot, label="Ground Truth")
    plt.plot(y_pred_plot, label="Prediction")

    # Add dashed separators between trials
    cumulative = 0
    for length in trial_lengths[:-1]:
        cumulative += length
        plt.axvline(x=cumulative, linestyle="--")

    plt.title(f"Patient {patient_id} - All validation trials")
    plt.xlabel("Time steps")
    plt.ylabel("Scaled GSR_mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_val_prediction_plots_for_fold(
    regressor: Sequential,
    val_path: str,
    fold_index: int,
) -> None:
    """Create per-patient prediction plots for all patients in data/val.

    Args:
        regressor: Trained Keras model.
        val_path: Folder containing external validation CSV files.
        fold_index: Current fold index used for output subfolder naming.
    """
    participant_to_files = get_participant_file_map(val_path)

    fold_plot_dir = os.path.join(PLOTS_DIR, f"fold_{fold_index}")
    os.makedirs(fold_plot_dir, exist_ok=True)

    for patient_id, patient_files in participant_to_files.items():
        output_path = os.path.join(
            fold_plot_dir, f"patient_{patient_id}_all_trials.png"
        )
        print(
            f"Creating validation prediction plot for patient {patient_id} "
            f"(fold {fold_index})..."
        )
        predict_val_patient_all_trials(
            regressor=regressor,
            patient_id=patient_id,
            patient_files=patient_files,
            output_path=output_path,
            target_column="GSR_mean",
        )


def main() -> None:
    """Run the full training, evaluation, and validation plotting pipeline."""
    ensure_output_structure()

    if not os.path.isdir(TRAIN_PATH):
        print(f"Train path not found: {TRAIN_PATH}")
        return

    if not os.path.isdir(VAL_PATH):
        print(f"Validation path not found: {VAL_PATH}")
        return

    print("Reading training data overview...")

    # Read all train CSVs once for a global overview like in the original script
    train_all_files = sorted(glob.glob(os.path.join(TRAIN_PATH, "*.csv")))
    if not train_all_files:
        print(f"No CSV files found in {TRAIN_PATH}")
        return

    li = []
    last_df = None
    for filename in train_all_files:
        df = load_csv(filename)
        li.append(df)
        last_df = df

    dataset = pd.concat(li, axis=0, ignore_index=True)
    dataset = dataset.fillna(0)

    print("Evaluating...")
    X = dataset.iloc[:, dataset.columns != "GSR_mean"]
    y = dataset.iloc[:, dataset.columns == "GSR_mean"].values

    print("X is :", X.shape)
    print("y is :", y.shape)

    print("Reading data for preprocessing...")
    participant_to_files_train = get_participant_file_map(TRAIN_PATH)

    participant_ids = sorted(participant_to_files_train.keys())
    print(f"Found {len(participant_ids)} participants in train.")

    list_of_lists = build_fold_lists(participant_to_files_train, n_splits=5)

    # Cross-validation loop
    for i in range(len(list_of_lists)):
        print("***********Splitting test and train...")

        train_lists_outer = list_of_lists[0:i] + list_of_lists[i + 1 :]
        test_list = list_of_lists[i]

        print("***********Splitting train and eval...")

        # Keep the original selection logic as closely as possible, but make it safe
        second_list_of_lists = train_lists_outer

        if not second_list_of_lists:
            print(f"Skipping fold {i} because no train/val folds remain.")
            continue

        if i < len(second_list_of_lists):
            val_idx = i
        else:
            val_idx = 0

        val_list = second_list_of_lists[val_idx]
        train_lists_inner = (
            second_list_of_lists[0:val_idx] + second_list_of_lists[val_idx + 1 :]
        )

        # Flatten training fold list
        train_dataframes: List[pd.DataFrame] = []
        for sublist in train_lists_inner:
            train_dataframes.extend(sublist)

        if not train_dataframes:
            print(f"Skipping fold {i} because training data is empty.")
            continue

        if not test_list:
            print(f"Skipping fold {i} because test data is empty.")
            continue

        if not val_list:
            print(f"Skipping fold {i} because validation data is empty.")
            continue

        # Prepare train data
        train_data = prepare_dataframe_list(train_dataframes)
        X_train, y_train = dataframe_to_xy(train_data, target_column="GSR_mean")
        X_train, y_train = scale_xy_separately(X_train, y_train)

        # Prepare test data
        test_data = prepare_dataframe_list(test_list)
        X_test, y_test = dataframe_to_xy(test_data, target_column="GSR_mean")
        X_test, y_test = scale_xy_separately(X_test, y_test)

        # Prepare val data
        val_data = prepare_dataframe_list(val_list)
        X_val, y_val = dataframe_to_xy(val_data, target_column="GSR_mean")
        X_val, y_val = scale_xy_separately(X_val, y_val)

        # Reshape for LSTM
        X_train = reshape_for_lstm(X_train)
        X_test = reshape_for_lstm(X_test)
        X_val = reshape_for_lstm(X_val)

        print("Shapes of X_train, X_test, and X_val...")
        print("X_train :", X_train.shape)
        print("X_test  :", X_test.shape)
        print("X_val   :", X_val.shape)

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

        # Fold test prediction + metrics
        y_pred = regressor.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)

        print("Results on Fold: " + str(i))
        print("Root mean squared error", rmse)

        nrmse_val = nrmse(rmse, y_test)
        print("Normalized root mean squared error", nrmse_val)

        row_contents = [str(i), str(rmse), str(nrmse_val)]
        append_list_as_row(RESULTS_CSV, row_contents)

        # Create validation prediction plots for all patients in data/val
        create_val_prediction_plots_for_fold(
            regressor=regressor,
            val_path=VAL_PATH,
            fold_index=i,
        )

    print("Done.")
    print(f"Results saved to: {RESULTS_CSV}")
    print(f"Prediction plots saved under: {PLOTS_DIR}")


if __name__ == "__main__":
    main()