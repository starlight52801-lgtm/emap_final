"""
Created on Mon Sep  6 20:48:16 2021

@author: harisushehu

Slight modifications of the original code were needed to run the code on the current data.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import time
from csv import writer
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyswarms as ps
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from joblib import Parallel, delayed


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "HeartRate_FS_DT",
        "output_root": "predictions/output/dt_reprod_hr",
        "random_state": 0,
    },
    "data": {
        "train_path": "data/train",
        "val_path": "data/val",
        "target_column": "heartrate_mean",
        "file_pattern": "*.csv",
        "filename_regex": r"Features_P(\d{3})-T(\d{2})\.csv",
        "encoding": "ISO-8859-1",
    },
    "model": {
        "type": "DecisionTreeRegressor",
        "max_depth": 20,
        "random_state": 0,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
    },
    "pso": {
        "enabled": True,
        "n_particles": 16,
        "iters": 25,
        "alpha": 0.88,
        "n_jobs": 32,
        "sample_train_rows": 20000,
        "sample_val_rows": 10000,
        "options": {
            "c1": 0.5,
            "c2": 0.5,
            "w": 0.9,
            "k": 12,
            "p": 2,
        },
    },
    "plots": {
        "dpi": 140,
        "figsize": [18, 6],
    },
}


def nrmse(rmse: float, y_true: np.ndarray) -> float:
    """Compute normalized root mean squared error.

    Args:
        rmse: Root mean squared error.
        y_true: Ground-truth target array in original scale.

    Returns:
        float: Normalized RMSE based on the range of y_true.
    """
    y_true = np.asarray(y_true).reshape(-1)
    value_range = float(np.max(y_true) - np.min(y_true))
    if value_range == 0.0:
        return float("nan")
    return float(rmse / value_range)


def append_list_as_row(file_name: str | Path, list_of_elem: List[Any]) -> None:
    """Append one row to a CSV file.

    Args:
        file_name: Path to the CSV file.
        list_of_elem: Row contents to append.
    """
    with open(file_name, "a+", newline="", encoding="utf-8") as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


def ensure_results_file(file_name: str | Path) -> None:
    """Create the main results CSV with header if it does not exist.

    Args:
        file_name: Path to the results CSV file.
    """
    if not os.path.exists(file_name):
        with open(file_name, "w", newline="", encoding="utf-8") as file_obj:
            header = [
                "Run",
                "Before_RMSE",
                "Before_NRMSE",
                "Before_R2",
                "After_RMSE",
                "After_NRMSE",
                "After_R2",
                "Selected_Features",
                "Total_Features",
                "PSO_Time_Seconds",
            ]
            filewriter = csv.DictWriter(file_obj, fieldnames=header)
            filewriter.writeheader()


def extract_patient_trial_info(filename: str, regex_pattern: str) -> Tuple[str, int]:
    """Extract patient and trial IDs from a filename.

    Args:
        filename: Basename of the CSV file.
        regex_pattern: Regular expression pattern with two groups:
            patient ID and trial ID.

    Returns:
        tuple[str, int]: Patient ID as string and trial ID as integer.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    match = re.match(regex_pattern, filename)
    if match is None:
        raise ValueError(
            f"Could not parse patient/trial from filename: {filename}"
        )

    patient_id = match.group(1)
    trial_id = int(match.group(2))
    return patient_id, trial_id


def create_output_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create the output directory structure for the current run.

    Args:
        config: Global configuration dictionary.

    Returns:
        dict[str, Path]: Dictionary containing important output paths.
    """
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_dir = Path(config["experiment"]["output_root"])
    experiment_dir = root_dir / config["experiment"]["name"]
    run_dir = experiment_dir / run_name
    plots_dir = run_dir / "plots_val_patients"
    artifacts_dir = run_dir / "artifacts"

    plots_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root_dir": root_dir,
        "experiment_dir": experiment_dir,
        "run_dir": run_dir,
        "plots_dir": plots_dir,
        "artifacts_dir": artifacts_dir,
        "results_csv": run_dir / "results.csv",
        "config_json": run_dir / "config.json",
        "selected_features_csv": run_dir / "selected_features.csv",
        "val_predictions_csv": run_dir / "val_predictions_all_rows.csv",
        "patient_metrics_csv": run_dir / "patient_metrics.csv",
        "model_joblib": artifacts_dir / "decision_tree_model.joblib",
        "scaler_x_joblib": artifacts_dir / "scaler_x.joblib",
        "scaler_y_joblib": artifacts_dir / "scaler_y.joblib",
    }


def read_csv_records(path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read CSV files with metadata from a directory.

    Each record contains:
    - filepath
    - filename
    - patient_id
    - trial_id
    - dataframe

    Args:
        path: Directory containing CSV files.
        config: Global configuration dictionary.

    Returns:
        list[dict[str, Any]]: List of file records with metadata.

    Raises:
        FileNotFoundError: If no CSV files are found in the given path.
    """
    file_pattern = config["data"]["file_pattern"]
    encoding = config["data"]["encoding"]
    regex_pattern = config["data"]["filename_regex"]

    all_files = sorted(glob.glob(os.path.join(path, file_pattern)))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in: {path}")

    records: List[Dict[str, Any]] = []
    for filepath in all_files:
        filename = os.path.basename(filepath)
        patient_id, trial_id = extract_patient_trial_info(filename, regex_pattern)
        df = pd.read_csv(filepath, encoding=encoding, header=0).fillna(0)

        records.append(
            {
                "filepath": filepath,
                "filename": filename,
                "patient_id": patient_id,
                "trial_id": trial_id,
                "dataframe": df,
            }
        )

    return records


def concatenate_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Concatenate all record dataframes into a single dataframe.

    Args:
        records: List of record dictionaries containing dataframes.

    Returns:
        pd.DataFrame: Concatenated dataframe.
    """
    return pd.concat([record["dataframe"] for record in records], axis=0, ignore_index=True)


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Split a dataframe into features and target.

    Args:
        df: Input dataframe.
        target_column: Name of the target column.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: Feature dataframe and target array.
    """
    X_df = df.drop(columns=[target_column])
    y = df[target_column].to_numpy().reshape(-1, 1)
    return X_df, y


def fit_scalers(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
) -> Tuple[StandardScaler, StandardScaler]:
    """Fit feature and target scalers on the training data.

    Args:
        X_train_df: Training features as dataframe.
        y_train: Training targets as array.

    Returns:
        tuple[StandardScaler, StandardScaler]: Fitted X and y scalers.
    """
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    scaler_x.fit(X_train_df)
    scaler_y.fit(y_train)

    return scaler_x, scaler_y


def transform_dataset(
    X_df: pd.DataFrame,
    y: np.ndarray,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a dataset using pre-fitted scalers.

    Args:
        X_df: Feature dataframe.
        y: Target array.
        scaler_x: Fitted feature scaler.
        scaler_y: Fitted target scaler.

    Returns:
        tuple[np.ndarray, np.ndarray]: Scaled features and scaled target.
    """
    X_scaled = scaler_x.transform(X_df)
    y_scaled = scaler_y.transform(y).reshape(-1)
    return X_scaled, y_scaled


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics in original scale.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        dict[str, float]: Dictionary with RMSE, NRMSE, and R2.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = float(sqrt(metrics.mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "nrmse": nrmse(rmse, y_true),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
) -> DecisionTreeRegressor:
    """Train a DecisionTreeRegressor from the config.

    Args:
        X_train: Scaled training features.
        y_train: Scaled training target as 1D array.
        config: Global configuration dictionary.

    Returns:
        DecisionTreeRegressor: Trained regressor.
    """
    reg = DecisionTreeRegressor(
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
        min_samples_split=config["model"]["min_samples_split"],
        min_samples_leaf=config["model"]["min_samples_leaf"],
    )
    reg.fit(X_train, y_train)
    return reg


def build_pso_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> Any:
    """Build the PSO objective function.

    The objective prefers high validation R2 and fewer selected features.
    Particle evaluations are parallelized across CPU cores.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        config: Global configuration dictionary.

    Returns:
        Callable: Objective function for pyswarms.
    """
    alpha = float(config["pso"]["alpha"])
    total_features = X_train.shape[1]
    n_jobs = int(config["pso"]["n_jobs"])

    def f_per_particle(mask_vector: np.ndarray) -> float:
        """Compute the objective value for one particle.

        Args:
            mask_vector: Particle position vector.

        Returns:
            float: Objective value to minimize.
        """
        selected_mask = mask_vector > 0.5

        # Avoid empty feature subsets.
        if np.count_nonzero(selected_mask) == 0:
            return 1e6

        X_subset = X_train[:, selected_mask]
        X_val_subset = X_val[:, selected_mask]

        # Train one tree for this particle.
        reg = train_decision_tree(X_subset, y_train, config)
        y_pred_val = reg.predict(X_val_subset)

        try:
            score_r2 = float(r2_score(y_val, y_pred_val))
        except Exception:
            score_r2 = -1.0

        # Lower is better for PSO.
        objective_value = (
            alpha * (1.0 - score_r2)
            + (1.0 - alpha) * (1.0 - (X_subset.shape[1] / total_features))
        )
        return float(objective_value)

    def objective_function(x: np.ndarray) -> np.ndarray:
        """Compute the objective value for the whole swarm in parallel.

        Args:
            x: Swarm positions of shape (n_particles, n_features).

        Returns:
            np.ndarray: Objective values for all particles.
        """
        scores = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(f_per_particle)(x[i]) for i in range(x.shape[0])
        )
        return np.array(scores, dtype=float)

    return objective_function

def subsample_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample rows without replacement if the dataset is larger than max_rows.

    Args:
        X: Feature matrix.
        y: Target vector.
        max_rows: Maximum number of rows to keep.
        random_state: Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray]: Subsampled X and y. If X already has
        at most max_rows rows, the original arrays are returned unchanged.
    """
    if max_rows is None or X.shape[0] <= max_rows:
        return X, y

    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], size=max_rows, replace=False)

    return X[indices], y[indices]

def run_pso_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    """Run PSO-based feature selection on subsampled data.

    The final selected mask is later used to train the model on the full
    training dataset. This keeps the PSO search much faster.

    Args:
        X_train: Full training features.
        y_train: Full training target.
        X_val: Full validation features.
        y_val: Full validation target.
        config: Global configuration dictionary.

    Returns:
        tuple[np.ndarray, float]: Best particle position and elapsed time.
    """
    dimensions = X_train.shape[1]
    options = config["pso"]["options"]

    # Use only subsets during PSO to speed up the search.
    X_train_pso, y_train_pso = subsample_rows(
        X=X_train,
        y=y_train,
        max_rows=config["pso"]["sample_train_rows"],
        random_state=config["experiment"]["random_state"],
    )
    X_val_pso, y_val_pso = subsample_rows(
        X=X_val,
        y=y_val,
        max_rows=config["pso"]["sample_val_rows"],
        random_state=config["experiment"]["random_state"] + 1,
    )

    print("PSO train subset shape:", X_train_pso.shape)
    print("PSO val subset shape  :", X_val_pso.shape)

    min_val = np.zeros(dimensions)
    max_val = np.ones(dimensions)
    values_bound = (min_val, max_val)

    objective_function = build_pso_objective(
        X_train=X_train_pso,
        y_train=y_train_pso,
        X_val=X_val_pso,
        y_val=y_val_pso,
        config=config,
    )

    optimizer = ps.single.GlobalBestPSO(
        n_particles=config["pso"]["n_particles"],
        dimensions=dimensions,
        options=options,
        bounds=values_bound,
    )

    start_time = time.time()
    _, best_position = optimizer.optimize(
        objective_function,
        iters=config["pso"]["iters"],
    )
    elapsed_time = time.time() - start_time

    return best_position, elapsed_time


def save_selected_features(
    feature_names: List[str],
    pos: np.ndarray,
    output_csv: str | Path,
) -> np.ndarray:
    """Save selected feature information to CSV.

    Args:
        feature_names: Full list of feature names.
        pos: Best PSO position vector.
        output_csv: Output CSV path.

    Returns:
        np.ndarray: Boolean mask of selected features.
    """
    selected_mask = pos > 0.5
    selected_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "position_value": pos,
            "selected": selected_mask.astype(int),
        }
    )
    selected_df.to_csv(output_csv, index=False)
    return selected_mask


def predict_original_scale(
    regressor: DecisionTreeRegressor,
    X_scaled: np.ndarray,
    scaler_y: StandardScaler,
) -> np.ndarray:
    """Predict target values and convert them back to original scale.

    Args:
        regressor: Trained regressor on scaled targets.
        X_scaled: Scaled input features.
        scaler_y: Fitted target scaler.

    Returns:
        np.ndarray: Predictions in original target scale.
    """
    y_pred_scaled = regressor.predict(X_scaled).reshape(-1, 1)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
    return y_pred_original


def build_val_prediction_table(
    val_records: List[Dict[str, Any]],
    feature_names: List[str],
    selected_mask: np.ndarray,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    regressor_before: DecisionTreeRegressor,
    regressor_after: DecisionTreeRegressor,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Build a row-level prediction table for all validation files.

    Args:
        val_records: Validation file records.
        feature_names: Full feature names in training order.
        selected_mask: Boolean mask of selected features.
        scaler_x: Fitted feature scaler.
        scaler_y: Fitted target scaler.
        regressor_before: Baseline model before feature selection.
        regressor_after: Final model after feature selection.
        config: Global configuration dictionary.

    Returns:
        pd.DataFrame: Row-level validation predictions with metadata.
    """
    target_column = config["data"]["target_column"]
    prediction_rows: List[pd.DataFrame] = []

    for record in sorted(val_records, key=lambda r: (r["patient_id"], r["trial_id"])):
        df_trial = record["dataframe"].copy()
        X_trial = df_trial[feature_names]
        y_trial = df_trial[target_column].to_numpy().reshape(-1, 1)

        X_trial_scaled = scaler_x.transform(X_trial)
        X_trial_selected = X_trial_scaled[:, selected_mask]

        y_pred_before = predict_original_scale(
            regressor=regressor_before,
            X_scaled=X_trial_scaled,
            scaler_y=scaler_y,
        )
        y_pred_after = predict_original_scale(
            regressor=regressor_after,
            X_scaled=X_trial_selected,
            scaler_y=scaler_y,
        )

        trial_df = pd.DataFrame(
            {
                "patient_id": record["patient_id"],
                "trial_id": record["trial_id"],
                "filename": record["filename"],
                "sample_idx_in_trial": np.arange(len(df_trial)),
                "y_true": y_trial.reshape(-1),
                "y_pred_before": y_pred_before,
                "y_pred_after": y_pred_after,
            }
        )
        prediction_rows.append(trial_df)

    return pd.concat(prediction_rows, axis=0, ignore_index=True)


def sanitize_filename(name: str) -> str:
    """Convert a free-form string into a filesystem-safe filename part.

    Args:
        name: Input string.

    Returns:
        str: Sanitized string.
    """
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name)


def plot_val_patients(
    prediction_df: pd.DataFrame,
    plots_dir: str | Path,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Create one concatenated plot per validation patient.

    All trials for a patient are concatenated in trial order. Dashed vertical
    lines separate individual trials.

    Args:
        prediction_df: Row-level validation predictions.
        plots_dir: Directory where plots will be saved.
        config: Global configuration dictionary.

    Returns:
        pd.DataFrame: Per-patient metrics table.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    figsize = tuple(config["plots"]["figsize"])
    dpi = int(config["plots"]["dpi"])

    patient_metrics: List[Dict[str, Any]] = []

    for patient_id, patient_df in prediction_df.groupby("patient_id"):
        patient_df = patient_df.sort_values(["trial_id", "sample_idx_in_trial"]).copy()

        x_values = np.arange(len(patient_df))
        y_true = patient_df["y_true"].to_numpy()
        y_pred_after = patient_df["y_pred_after"].to_numpy()

        # Compute boundaries between trials.
        trial_sizes = (
            patient_df.groupby("trial_id", sort=True)
            .size()
            .reset_index(name="n_rows")
        )
        boundaries = np.cumsum(trial_sizes["n_rows"].to_numpy())[:-1]

        metrics_patient = compute_metrics(y_true=y_true, y_pred=y_pred_after)
        patient_metrics.append(
            {
                "patient_id": patient_id,
                "rmse": metrics_patient["rmse"],
                "nrmse": metrics_patient["nrmse"],
                "r2": metrics_patient["r2"],
                "n_samples": len(patient_df),
                "n_trials": int(patient_df["trial_id"].nunique()),
            }
        )

        plt.figure(figsize=figsize)
        plt.plot(x_values, y_true, label="Ground truth", linewidth=1.5)
        plt.plot(x_values, y_pred_after, label="Prediction", linewidth=1.2)

        for boundary in boundaries:
            plt.axvline(boundary, linestyle="--", linewidth=0.8)

        plt.title(
            f"Validation patient P{patient_id} | "
            f"Trials: {patient_df['trial_id'].nunique()} | "
            f"RMSE: {metrics_patient['rmse']:.3f} | "
            f"R2: {metrics_patient['r2']:.3f}"
        )
        plt.xlabel("Concatenated sample index")
        plt.ylabel("Heart rate")
        plt.legend()
        plt.tight_layout()

        output_path = plots_dir / f"patient_P{sanitize_filename(patient_id)}.png"
        plt.savefig(output_path, dpi=dpi)
        plt.close()

    return pd.DataFrame(patient_metrics)


def main() -> None:
    """Run the full experiment."""
    output_paths = create_output_dirs(CONFIG)

    # Save the exact run config.
    with open(output_paths["config_json"], "w", encoding="utf-8") as file_obj:
        json.dump(CONFIG, file_obj, indent=4)

    ensure_results_file(output_paths["results_csv"])

    print("Reading training data...")
    train_records = read_csv_records(CONFIG["data"]["train_path"], CONFIG)

    print("Reading validation data...")
    val_records = read_csv_records(CONFIG["data"]["val_path"], CONFIG)

    print("Number of train files:", len(train_records))
    print("Number of val files:", len(val_records))
    print(
        "Detected train participants:",
        len(sorted({record["patient_id"] for record in train_records})),
    )
    print(
        "Detected val participants:",
        len(sorted({record["patient_id"] for record in val_records})),
    )

    train_df = concatenate_records(train_records).fillna(0)
    val_df = concatenate_records(val_records).fillna(0)

    target_column = CONFIG["data"]["target_column"]

    X_train_df, y_train = split_features_target(train_df, target_column)
    X_val_df, y_val = split_features_target(val_df, target_column)

    feature_names = list(X_train_df.columns)

    print("Train X is :", X_train_df.shape)
    print("Train y is :", y_train.shape)
    print("Val X is   :", X_val_df.shape)
    print("Val y is   :", y_val.shape)

    # Fit scalers on train only and apply to both splits.
    scaler_x, scaler_y = fit_scalers(X_train_df, y_train)
    X_train_scaled, y_train_scaled = transform_dataset(
        X_train_df, y_train, scaler_x, scaler_y
    )
    X_val_scaled, y_val_scaled = transform_dataset(
        X_val_df, y_val, scaler_x, scaler_y
    )

    # -----------------------------------------------------------------
    # Baseline before feature selection
    # -----------------------------------------------------------------
    print("Before feature selection...")
    reg_before = train_decision_tree(X_train_scaled, y_train_scaled, CONFIG)
    y_pred_before = predict_original_scale(reg_before, X_val_scaled, scaler_y)
    metrics_before = compute_metrics(y_true=y_val.reshape(-1), y_pred=y_pred_before)

    print("Before RMSE :", metrics_before["rmse"])
    print("Before NRMSE:", metrics_before["nrmse"])
    print("Before R2   :", metrics_before["r2"])

    # -----------------------------------------------------------------
    # Feature selection
    # -----------------------------------------------------------------
    if CONFIG["pso"]["enabled"]:
        print("Running PSO feature selection...")
        best_pos, pso_elapsed_seconds = run_pso_feature_selection(
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            X_val=X_val_scaled,
            y_val=y_val_scaled,
            config=CONFIG,
        )
    else:
        print("PSO feature selection disabled. Using all features.")
        best_pos = np.ones(X_train_scaled.shape[1], dtype=float)
        pso_elapsed_seconds = 0.0

    selected_mask = save_selected_features(
        feature_names=feature_names,
        pos=best_pos,
        output_csv=output_paths["selected_features_csv"],
    )

    n_selected_features = int(np.count_nonzero(selected_mask))
    if n_selected_features == 0:
        raise ValueError("PSO selected zero features. Aborting.")

    print("Selected features:", n_selected_features, "/", len(feature_names))
    print("PSO time (seconds):", pso_elapsed_seconds)

    X_train_selected = X_train_scaled[:, selected_mask]
    X_val_selected = X_val_scaled[:, selected_mask]

    # -----------------------------------------------------------------
    # Final model after feature selection
    # -----------------------------------------------------------------
    print("After feature selection...")
    reg_after = train_decision_tree(X_train_selected, y_train_scaled, CONFIG)
    y_pred_after = predict_original_scale(reg_after, X_val_selected, scaler_y)
    metrics_after = compute_metrics(y_true=y_val.reshape(-1), y_pred=y_pred_after)

    print("After RMSE :", metrics_after["rmse"])
    print("After NRMSE:", metrics_after["nrmse"])
    print("After R2   :", metrics_after["r2"])

    # -----------------------------------------------------------------
    # Save results and artifacts
    # -----------------------------------------------------------------
    row_contents = [
        output_paths["run_dir"].name,
        metrics_before["rmse"],
        metrics_before["nrmse"],
        metrics_before["r2"],
        metrics_after["rmse"],
        metrics_after["nrmse"],
        metrics_after["r2"],
        n_selected_features,
        len(feature_names),
        pso_elapsed_seconds,
    ]
    append_list_as_row(output_paths["results_csv"], row_contents)

    joblib.dump(reg_after, output_paths["model_joblib"])
    joblib.dump(scaler_x, output_paths["scaler_x_joblib"])
    joblib.dump(scaler_y, output_paths["scaler_y_joblib"])

    # -----------------------------------------------------------------
    # Save validation predictions and patient-wise plots
    # -----------------------------------------------------------------
    print("Creating validation predictions and patient plots...")
    prediction_df = build_val_prediction_table(
        val_records=val_records,
        feature_names=feature_names,
        selected_mask=selected_mask,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        regressor_before=reg_before,
        regressor_after=reg_after,
        config=CONFIG,
    )
    prediction_df.to_csv(output_paths["val_predictions_csv"], index=False)

    patient_metrics_df = plot_val_patients(
        prediction_df=prediction_df,
        plots_dir=output_paths["plots_dir"],
        config=CONFIG,
    )
    patient_metrics_df.to_csv(output_paths["patient_metrics_csv"], index=False)

    print("Done.")
    print("Run directory:", output_paths["run_dir"])


if __name__ == "__main__":
    main()