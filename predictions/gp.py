"""
Temporal GP regressor for EMAP CSV files using gplearn.

Main ideas:
- Train only on `data/train`.
- Evaluate and plot on `data/val`.
- Keep all EEG_* features.
- Always include IRPleth_mean and Respir_mean.
- Add temporal context from IRPleth / Respir:
    - lag values
    - rolling mean
    - rolling std
    - slope-like differences
- Use a custom GP fitness that discourages degenerate near-constant solutions.
- Optionally apply a linear calibration on top of the raw GP output.
- Save metrics, predictions, patient plots, convergence plot, and program text
  into a run-specific output folder.

Targets:
- "heartrate"
- "gsr"
- "arousal"

Notes:
- GPU is not used here. gplearn is CPU-based.
- Parallelization is handled via `n_jobs`.
- The final RMSE / NRMSE are always computed on the original target scale.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor


# =============================================================================
# User configuration
# =============================================================================

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
TRAIN_DIR = Path("./data/train")
VAL_DIR = Path("./data/val")
OUTPUT_ROOT = Path("./predictions/output/gp")

# -------------------------------------------------------------------------
# Target selection
# Allowed:
#   "heartrate"
#   "gsr"
#   "arousal"
# -------------------------------------------------------------------------
TARGET_NAME = "gsr"

# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
RANDOM_SEED = 42

# -------------------------------------------------------------------------
# Parallelization
# -------------------------------------------------------------------------
N_JOBS = 60

# -------------------------------------------------------------------------
# GP search configuration
# -------------------------------------------------------------------------
POPULATION_SIZE = 4000
GENERATIONS = 50
TOURNAMENT_SIZE = 20
STOPPING_CRITERIA = 0.0
INIT_DEPTH = (2, 6)
INIT_METHOD = "half and half"

P_CROSSOVER = 0.70
P_SUBTREE_MUTATION = 0.10
P_HOIST_MUTATION = 0.05
P_POINT_MUTATION = 0.10
P_POINT_REPLACE = 0.05

PARSIMONY_COEFFICIENT = 0.0005
MAX_SAMPLES = 1.0
VERBOSE = 1

# -------------------------------------------------------------------------
# Temporal feature engineering
# -------------------------------------------------------------------------
# Number of previous timesteps to use for IRPleth / Respir lags.
N_LAGS = 10

# Use rolling summary features over the previous N_LAGS timesteps.
USE_ROLLING_FEATURES = True

# Add simple first-order temporal change features.
USE_DIFF_FEATURES = True

# -------------------------------------------------------------------------
# Optional train subsampling for speed
# -------------------------------------------------------------------------
USE_BALANCED_TRIAL_SAMPLING = True
MAX_ROWS_PER_TRIAL_FOR_TRAINING = 300

# -------------------------------------------------------------------------
# Model stabilization options
# -------------------------------------------------------------------------
# Custom fitness weights to reduce collapse to near-constant predictions.
VARIANCE_PENALTY_WEIGHT = 0.15
CORRELATION_PENALTY_WEIGHT = 2.0

HR_MIN = 35.0
HR_MAX = 220.0
RANGE_PENALTY_WEIGHT = 2.0


# After GP training, fit y ≈ a * pred + b on full train and apply it to
# both train and val predictions.
USE_LINEAR_CALIBRATION = True

# -------------------------------------------------------------------------
# Data / feature configuration
# -------------------------------------------------------------------------
EEG_PREFIX = "EEG_"
ALWAYS_INCLUDE_PHYSIO = ["IRPleth_mean", "Respir_mean"]

TARGET_COLUMN_MAP = {
    "heartrate": "heartrate_mean",
    "gsr": "GSR_mean",
    "arousal": "LABEL_SR_Arousal",
}

ALL_TARGET_COLUMNS = [
    "heartrate_mean",
    "GSR_mean",
    "LABEL_SR_Arousal",
]

# -------------------------------------------------------------------------
# Output controls
# -------------------------------------------------------------------------
SAVE_TRAIN_PREDICTIONS = True
SAVE_VAL_PATIENT_PLOTS = True
SAVE_VAL_OVERVIEW_PLOT = True
SAVE_SCATTER_PLOT = True
SAVE_RESIDUAL_HIST = True


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class TrialData:
    """Container for one trial file.

    Attributes:
        filepath: Original CSV file path.
        patient_id: Zero-padded patient ID, e.g. "001".
        trial_id: Integer trial ID.
        df: Loaded DataFrame for this trial.
    """

    filepath: Path
    patient_id: str
    trial_id: int
    df: pd.DataFrame


@dataclass
class SplitData:
    """Prepared split data after loading and feature engineering.

    Attributes:
        X: Feature matrix.
        y: Target vector in original scale.
        row_index_df: Metadata for each row.
        trials: Original trial list.
        feature_names: Ordered feature names used in X.
        target_column: Target column name used for y.
    """

    X: np.ndarray
    y: np.ndarray
    row_index_df: pd.DataFrame
    trials: List[TrialData]
    feature_names: List[str]
    target_column: str


# =============================================================================
# Utility functions
# =============================================================================

def create_run_dir(output_root: Path, target_name: str) -> Path:
    """Create a run-specific output directory.

    Args:
        output_root: Base output directory.
        target_name: Selected logical target name.

    Returns:
        Created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{target_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(filepath: Path, payload: Dict) -> None:
    """Save a dictionary as JSON.

    Args:
        filepath: Output file path.
        payload: JSON-serializable dictionary.
    """
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE value.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute normalized RMSE based on the target range.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        NRMSE value. Returns NaN if the target range is zero.
    """
    y_true = np.asarray(y_true).reshape(-1)
    score_rmse = rmse(y_true, y_pred)
    target_range = float(np.max(y_true) - np.min(y_true))
    if target_range == 0.0:
        return float("nan")
    return float(score_rmse / target_range)


def parse_patient_and_trial(filepath: Path) -> Tuple[str, int]:
    """Extract patient and trial ID from filename.

    Args:
        filepath: CSV file path.

    Returns:
        Tuple of (patient_id, trial_id).

    Raises:
        ValueError: If parsing fails.
    """
    filename = filepath.name
    match = re.search(r"P(\d+).*?T(\d+)", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Could not parse patient/trial from filename: {filename}. "
            "Expected something like P001 and T01."
        )

    patient_id = match.group(1).zfill(3)
    trial_id = int(match.group(2))
    return patient_id, trial_id


def discover_csv_files(folder: Path) -> List[Path]:
    """Find all CSV files in a folder.

    Args:
        folder: Folder path.

    Returns:
        Sorted list of CSV paths.
    """
    return sorted(folder.glob("*.csv"))


def load_trials(folder: Path) -> List[TrialData]:
    """Load all CSV trials from a folder.

    Args:
        folder: Directory containing CSV files.

    Returns:
        List of TrialData objects.

    Raises:
        FileNotFoundError: If no CSV files are found.
    """
    csv_files = discover_csv_files(folder)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    trials: List[TrialData] = []
    for filepath in csv_files:
        patient_id, trial_id = parse_patient_and_trial(filepath)
        df = pd.read_csv(filepath, encoding="ISO-8859-1", header=0).fillna(0)
        trials.append(
            TrialData(
                filepath=filepath,
                patient_id=patient_id,
                trial_id=trial_id,
                df=df,
            )
        )
    return trials


def determine_base_feature_columns(example_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Determine the input feature columns based on the current rules.

    Rules:
    - include all columns starting with EEG_
    - always include IRPleth_mean and Respir_mean if present
    - never include heartrate_mean, GSR_mean, LABEL_SR_Arousal as inputs

    Args:
        example_df: Example DataFrame used to inspect available columns.

    Returns:
        Tuple of:
            - eeg_columns
            - base_feature_columns
    """
    eeg_columns = sorted([col for col in example_df.columns if col.startswith(EEG_PREFIX)])

    missing_required = [col for col in ALWAYS_INCLUDE_PHYSIO if col not in example_df.columns]
    if missing_required:
        raise KeyError(
            f"Missing required always-include physio columns: {missing_required}"
        )

    base_feature_columns = eeg_columns + ALWAYS_INCLUDE_PHYSIO
    base_feature_columns = [
        col for col in base_feature_columns if col not in ALL_TARGET_COLUMNS
    ]

    if not base_feature_columns:
        raise ValueError("No usable input features were found.")

    return eeg_columns, base_feature_columns


def add_temporal_features_to_trial(
    df: pd.DataFrame,
    eeg_columns: List[str],
    base_feature_columns: List[str],
) -> pd.DataFrame:
    """Add temporal context features to one trial DataFrame.

    The design keeps all EEG columns as current-step inputs and adds temporal
    context mainly through IRPleth / Respir features. This keeps the total
    dimensionality manageable while still giving the model sequence context.

    Added features:
    - current EEG_* and current IRPleth / Respir
    - lags for IRPleth / Respir
    - rolling mean / std for IRPleth / Respir
    - first-order differences for IRPleth / Respir

    Args:
        df: Trial DataFrame.
        eeg_columns: All EEG_* columns.
        base_feature_columns: Current-step feature columns.

    Returns:
        Trial DataFrame with engineered features and dropped invalid early rows.
    """
    work_df = df.copy()

    # Keep current-step raw inputs.
    feature_df = work_df[base_feature_columns].copy()

    for phys_col in ALWAYS_INCLUDE_PHYSIO:
        # Add lag features for temporal context.
        for lag in range(1, N_LAGS + 1):
            feature_df[f"{phys_col}_lag_{lag}"] = work_df[phys_col].shift(lag)

        if USE_ROLLING_FEATURES:
            # Rolling summary features over the recent history.
            feature_df[f"{phys_col}_rollmean_{N_LAGS}"] = (
                work_df[phys_col].shift(1).rolling(window=N_LAGS).mean()
            )
            feature_df[f"{phys_col}_rollstd_{N_LAGS}"] = (
                work_df[phys_col].shift(1).rolling(window=N_LAGS).std()
            )

        if USE_DIFF_FEATURES:
            # Simple temporal change features.
            feature_df[f"{phys_col}_diff_1"] = work_df[phys_col].diff(1)
            feature_df[f"{phys_col}_diff_2"] = work_df[phys_col].diff(2)
            feature_df[f"{phys_col}_diff_prev"] = (
                work_df[phys_col].shift(1) - work_df[phys_col].shift(2)
            )

    # Add a compact EEG summary so the GP can also use coarse EEG dynamics
    # without having to build it from hundreds of columns every time.
    feature_df["EEG_global_mean"] = work_df[eeg_columns].mean(axis=1)
    feature_df["EEG_global_std"] = work_df[eeg_columns].std(axis=1)

    if USE_ROLLING_FEATURES:
        feature_df[f"EEG_global_mean_rollmean_{N_LAGS}"] = (
            feature_df["EEG_global_mean"].shift(1).rolling(window=N_LAGS).mean()
        )
        feature_df[f"EEG_global_mean_rollstd_{N_LAGS}"] = (
            feature_df["EEG_global_mean"].shift(1).rolling(window=N_LAGS).std()
        )

    if USE_DIFF_FEATURES:
        feature_df["EEG_global_mean_diff_1"] = feature_df["EEG_global_mean"].diff(1)

    # Replace NaNs introduced by early shifts/rolling and then drop rows that
    # still do not have a full temporal history.
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    min_required_history = N_LAGS
    valid_mask = np.arange(len(feature_df)) >= min_required_history
    feature_df = feature_df.loc[valid_mask].copy()
    feature_df = feature_df.fillna(0.0)

    return feature_df


def build_split_data(trials: List[TrialData], target_name: str) -> SplitData:
    """Build X, y and metadata for a split.

    Args:
        trials: Loaded trial files.
        target_name: Logical target name.

    Returns:
        SplitData object.
    """
    target_column = TARGET_COLUMN_MAP[target_name]
    eeg_columns, base_feature_columns = determine_base_feature_columns(trials[0].df)

    all_x_parts: List[np.ndarray] = []
    all_y_parts: List[np.ndarray] = []
    row_meta: List[pd.DataFrame] = []
    feature_names: List[str] | None = None

    for trial in trials:
        df = trial.df.copy()

        if target_column not in df.columns:
            raise KeyError(
                f"Target column '{target_column}' missing in file: {trial.filepath}"
            )

        feature_df = add_temporal_features_to_trial(
            df=df,
            eeg_columns=eeg_columns,
            base_feature_columns=base_feature_columns,
        )

        # Align target and metadata with the rows that remain after temporal
        # feature generation.
        aligned_target = df[target_column].iloc[len(df) - len(feature_df):].to_numpy(dtype=np.float64)

        if feature_names is None:
            feature_names = feature_df.columns.tolist()

        x_part = feature_df.to_numpy(dtype=np.float64)
        y_part = aligned_target

        all_x_parts.append(x_part)
        all_y_parts.append(y_part)

        meta_df = pd.DataFrame(
            {
                "patient_id": [trial.patient_id] * len(feature_df),
                "trial_id": [trial.trial_id] * len(feature_df),
                "filepath": [str(trial.filepath)] * len(feature_df),
                "row_in_trial": np.arange(len(feature_df), dtype=int),
            }
        )
        row_meta.append(meta_df)

    X = np.vstack(all_x_parts)
    y = np.concatenate(all_y_parts)
    row_index_df = pd.concat(row_meta, axis=0, ignore_index=True)

    return SplitData(
        X=X,
        y=y,
        row_index_df=row_index_df,
        trials=trials,
        feature_names=feature_names if feature_names is not None else [],
        target_column=target_column,
    )


def zscore_fit_transform(
    X_train: np.ndarray,
    X_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit z-score normalization on train and apply to another split.

    Args:
        X_train: Train feature matrix.
        X_other: Other split feature matrix.

    Returns:
        Tuple:
            X_train_scaled,
            X_other_scaled,
            mean,
            std
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_other_scaled = (X_other - mean) / std
    return X_train_scaled, X_other_scaled, mean, std


def build_balanced_trial_training_subset(
    split_data: SplitData,
    max_rows_per_trial: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a balanced subset across trials for faster GP training.

    Args:
        split_data: Training split data.
        max_rows_per_trial: Maximum number of rows to sample per trial.

    Returns:
        Tuple of (X_subset, y_subset).
    """
    meta = split_data.row_index_df.copy()
    meta["global_row_idx"] = np.arange(len(meta), dtype=int)

    sampled_indices: List[int] = []
    grouped = meta.groupby(["patient_id", "trial_id"], sort=True)

    for (_, _), group_df in grouped:
        indices = group_df["global_row_idx"].to_numpy()
        if len(indices) <= max_rows_per_trial:
            sampled = indices
        else:
            sampled = np.random.choice(indices, size=max_rows_per_trial, replace=False)
        sampled_indices.extend(sampled.tolist())

    sampled_indices = np.array(sorted(sampled_indices), dtype=int)
    return split_data.X[sampled_indices], split_data.y[sampled_indices]


def fit_linear_calibration(y_true: np.ndarray, y_pred_raw: np.ndarray) -> Tuple[float, float]:
    """Fit a linear calibration y ≈ a * pred + b.

    Args:
        y_true: Ground truth values.
        y_pred_raw: Raw model predictions.

    Returns:
        Tuple of (a, b).
    """
    x = np.asarray(y_pred_raw).reshape(-1)
    y = np.asarray(y_true).reshape(-1)

    # Solve least squares for y = a*x + b.
    design = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(a), float(b)


def apply_linear_calibration(y_pred_raw: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply linear calibration to predictions.

    Args:
        y_pred_raw: Raw model predictions.
        a: Slope.
        b: Intercept.

    Returns:
        Calibrated predictions.
    """
    return a * np.asarray(y_pred_raw).reshape(-1) + b


def save_metrics_csv(filepath: Path, rows: List[Dict[str, float]]) -> None:
    """Save metrics rows to CSV.

    Args:
        filepath: Output file path.
        rows: List of row dictionaries.
    """
    pd.DataFrame(rows).to_csv(filepath, index=False)


def create_prediction_table(
    split_data: SplitData,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Create row-level prediction table.

    Args:
        split_data: Split data with metadata.
        y_pred: Predictions aligned with split rows.

    Returns:
        DataFrame containing metadata, y_true and y_pred.
    """
    pred_df = split_data.row_index_df.copy()
    pred_df["y_true"] = split_data.y
    pred_df["y_pred"] = np.asarray(y_pred).reshape(-1)
    return pred_df


# =============================================================================
# Custom GP functions
# =============================================================================

def _protected_division(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Perform protected division.

    Args:
        x1: Numerator array.
        x2: Denominator array.

    Returns:
        Safely divided array.
    """
    return np.where(np.abs(x2) > 1e-8, x1 / x2, 1.0)


def _protected_log(x1: np.ndarray) -> np.ndarray:
    """Perform protected logarithm.

    Args:
        x1: Input array.

    Returns:
        log(1 + |x|) array.
    """
    return np.log1p(np.abs(x1))


def _protected_sqrt(x1: np.ndarray) -> np.ndarray:
    """Perform protected square root.

    Args:
        x1: Input array.

    Returns:
        sqrt(|x|) array.
    """
    return np.sqrt(np.abs(x1))


def _protected_inv(x1: np.ndarray) -> np.ndarray:
    """Perform protected inverse.

    Args:
        x1: Input array.

    Returns:
        Safe inverse array.
    """
    return np.where(np.abs(x1) > 1e-8, 1.0 / x1, 1.0)


PROTECTED_DIV = make_function(function=_protected_division, name="div", arity=2)
PROTECTED_LOG = make_function(function=_protected_log, name="log1pabs", arity=1)
PROTECTED_SQRT = make_function(function=_protected_sqrt, name="sqrtabs", arity=1)
PROTECTED_INV = make_function(function=_protected_inv, name="inv", arity=1)


# =============================================================================
# Custom fitness
# =============================================================================

def _stabilized_gp_fitness(
    y: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    """Compute a stabilized GP fitness.

    The goal is to reduce collapse to a nearly constant predictor.

    Objective minimized internally:
        RMSE
        + variance penalty
        + correlation penalty

    Since gplearn maximizes, this function returns the negative objective.

    Args:
        y: Ground truth values.
        y_pred: Predicted values.
        sample_weight: Sample weights.

    Returns:
        Negative objective value.
    """
    del sample_weight

    y = np.asarray(y).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    current_rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    std_true = float(np.std(y))
    std_pred = float(np.std(y_pred))

    if std_true < 1e-12:
        variance_penalty = 0.0
    else:
        variance_penalty = abs(std_pred - std_true) / std_true

    if std_true < 1e-12 or std_pred < 1e-12:
        corr_penalty = 1.0
    else:
        corr = float(np.corrcoef(y, y_pred)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        corr_penalty = 1.0 - max(corr, 0.0)

    if TARGET_NAME == "heartrate":
        below = np.maximum(HR_MIN - y_pred, 0.0)
        above = np.maximum(y_pred - HR_MAX, 0.0)
        range_penalty = float(np.mean(below + above))

        objective = (
            current_rmse
            + VARIANCE_PENALTY_WEIGHT * variance_penalty
            + CORRELATION_PENALTY_WEIGHT * corr_penalty
            + RANGE_PENALTY_WEIGHT * range_penalty
        )
    else:
        objective = (
            current_rmse
            + VARIANCE_PENALTY_WEIGHT * variance_penalty
            + CORRELATION_PENALTY_WEIGHT * corr_penalty
        )
    return -objective


STABILIZED_FITNESS = make_fitness(
    function=_stabilized_gp_fitness,
    greater_is_better=True,
)


# =============================================================================
# Plot helpers
# =============================================================================

def plot_convergence(run_details: pd.DataFrame, output_path: Path) -> None:
    """Plot training convergence curves.

    Args:
        run_details: DataFrame built from gplearn run_details_.
        output_path: Output image path.
    """
    plt.figure(figsize=(12, 6))

    if "average_fitness" in run_details.columns:
        # The fitness is negative objective, so multiply by -1 for display.
        plt.plot(-run_details["average_fitness"].to_numpy(), label="Average objective")
    if "best_fitness" in run_details.columns:
        plt.plot(-run_details["best_fitness"].to_numpy(), label="Best objective")

    plt.xlabel("Generation")
    plt.ylabel("Objective")
    plt.title("GP convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_val_patients(
    pred_df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
) -> None:
    """Create one concatenated prediction plot per validation patient.

    Trials are ordered by trial_id and separated by dashed vertical lines.

    Args:
        pred_df: Validation prediction table.
        target_column: Target column name for axis labeling.
        output_dir: Directory where plots should be stored.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for patient_id, patient_df in pred_df.groupby("patient_id", sort=True):
        patient_df = patient_df.sort_values(["trial_id", "row_in_trial"]).reset_index(drop=True)

        y_true = patient_df["y_true"].to_numpy()
        y_pred = patient_df["y_pred"].to_numpy()

        boundary_positions: List[int] = []
        cumulative = 0
        for _, trial_df in patient_df.groupby("trial_id", sort=True):
            cumulative += len(trial_df)
            boundary_positions.append(cumulative)

        plt.figure(figsize=(18, 6))
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Pred")

        for boundary in boundary_positions[:-1]:
            plt.axvline(boundary, linestyle="--", linewidth=1)

        plt.title(f"VAL - Patient {patient_id} - All Trials")
        plt.xlabel("Concatenated time steps across trials")
        plt.ylabel(target_column)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"patient_{patient_id}_val_true_vs_pred.png", dpi=150)
        plt.close()


def plot_val_overview(
    pred_df: pd.DataFrame,
    target_column: str,
    output_path: Path,
) -> None:
    """Create one global validation overview plot across all patients.

    Args:
        pred_df: Validation prediction DataFrame.
        target_column: Target column name.
        output_path: Output image path.
    """
    ordered = pred_df.sort_values(["patient_id", "trial_id", "row_in_trial"]).reset_index(drop=True)

    y_true = ordered["y_true"].to_numpy()
    y_pred = ordered["y_pred"].to_numpy()

    boundary_positions: List[int] = []
    cumulative = 0
    for _, trial_df in ordered.groupby(["patient_id", "trial_id"], sort=True):
        cumulative += len(trial_df)
        boundary_positions.append(cumulative)

    plt.figure(figsize=(20, 7))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")

    for boundary in boundary_positions[:-1]:
        plt.axvline(boundary, linestyle="--", linewidth=0.8)

    plt.title("VAL - All Patients / All Trials")
    plt.xlabel("Concatenated time steps")
    plt.ylabel(target_column)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_true_vs_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_column: str,
    output_path: Path,
) -> None:
    """Create a scatter plot of true vs predicted values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        target_column: Target column name.
        output_path: Output image path.
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.25, s=8)

    data_min = float(min(np.min(y_true), np.min(y_pred)))
    data_max = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([data_min, data_max], [data_min, data_max])

    plt.xlabel(f"True {target_column}")
    plt.ylabel(f"Pred {target_column}")
    plt.title("True vs Pred")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Create a histogram of residuals.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        output_path: Output image path.
    """
    residuals = np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=60)
    plt.xlabel("Residual (true - pred)")
    plt.ylabel("Count")
    plt.title("Residual histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run the full temporal GP workflow."""
    if TARGET_NAME not in TARGET_COLUMN_MAP:
        raise ValueError(
            f"Unsupported TARGET_NAME='{TARGET_NAME}'. "
            f"Allowed: {list(TARGET_COLUMN_MAP.keys())}"
        )

    np.random.seed(RANDOM_SEED)

    run_dir = create_run_dir(OUTPUT_ROOT, TARGET_NAME)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train trials...")
    train_trials = load_trials(TRAIN_DIR)

    print("Loading val trials...")
    val_trials = load_trials(VAL_DIR)

    print("Building temporal split data...")
    train_data = build_split_data(train_trials, TARGET_NAME)
    val_data = build_split_data(val_trials, TARGET_NAME)

    if train_data.feature_names != val_data.feature_names:
        raise ValueError("Train and val feature columns do not match.")

    # Fit scaling on train only and apply it to validation.
    X_train_scaled, X_val_scaled, train_mean, train_std = zscore_fit_transform(
        train_data.X,
        val_data.X,
    )

    train_data = SplitData(
        X=X_train_scaled,
        y=train_data.y.copy(),
        row_index_df=train_data.row_index_df,
        trials=train_data.trials,
        feature_names=train_data.feature_names,
        target_column=train_data.target_column,
    )
    val_data = SplitData(
        X=X_val_scaled,
        y=val_data.y.copy(),
        row_index_df=val_data.row_index_df,
        trials=val_data.trials,
        feature_names=val_data.feature_names,
        target_column=val_data.target_column,
    )

    if USE_BALANCED_TRIAL_SAMPLING:
        X_train_fit, y_train_fit = build_balanced_trial_training_subset(
            train_data,
            max_rows_per_trial=MAX_ROWS_PER_TRIAL_FOR_TRAINING,
        )
    else:
        X_train_fit, y_train_fit = train_data.X, train_data.y

    print("Train shape:", train_data.X.shape, train_data.y.shape)
    print("Val shape:  ", val_data.X.shape, val_data.y.shape)
    print("Training subset shape:", X_train_fit.shape, y_train_fit.shape)
    print("Number of features:", len(train_data.feature_names))
    print("Target column:", train_data.target_column)
    print("Parallel n_jobs:", N_JOBS)

    run_config = {
        "TRAIN_DIR": str(TRAIN_DIR),
        "VAL_DIR": str(VAL_DIR),
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "TARGET_NAME": TARGET_NAME,
        "TARGET_COLUMN": train_data.target_column,
        "RANDOM_SEED": RANDOM_SEED,
        "N_JOBS": N_JOBS,
        "POPULATION_SIZE": POPULATION_SIZE,
        "GENERATIONS": GENERATIONS,
        "TOURNAMENT_SIZE": TOURNAMENT_SIZE,
        "STOPPING_CRITERIA": STOPPING_CRITERIA,
        "INIT_DEPTH": list(INIT_DEPTH),
        "INIT_METHOD": INIT_METHOD,
        "P_CROSSOVER": P_CROSSOVER,
        "P_SUBTREE_MUTATION": P_SUBTREE_MUTATION,
        "P_HOIST_MUTATION": P_HOIST_MUTATION,
        "P_POINT_MUTATION": P_POINT_MUTATION,
        "P_POINT_REPLACE": P_POINT_REPLACE,
        "PARSIMONY_COEFFICIENT": PARSIMONY_COEFFICIENT,
        "MAX_SAMPLES": MAX_SAMPLES,
        "N_LAGS": N_LAGS,
        "USE_ROLLING_FEATURES": USE_ROLLING_FEATURES,
        "USE_DIFF_FEATURES": USE_DIFF_FEATURES,
        "USE_BALANCED_TRIAL_SAMPLING": USE_BALANCED_TRIAL_SAMPLING,
        "MAX_ROWS_PER_TRIAL_FOR_TRAINING": MAX_ROWS_PER_TRIAL_FOR_TRAINING,
        "VARIANCE_PENALTY_WEIGHT": VARIANCE_PENALTY_WEIGHT,
        "CORRELATION_PENALTY_WEIGHT": CORRELATION_PENALTY_WEIGHT,
        "USE_LINEAR_CALIBRATION": USE_LINEAR_CALIBRATION,
        "N_FEATURES": len(train_data.feature_names),
        "FEATURE_NAMES": train_data.feature_names,
        "TRAIN_ROWS_FULL": int(len(train_data.y)),
        "TRAIN_ROWS_FIT": int(len(y_train_fit)),
        "VAL_ROWS_FULL": int(len(val_data.y)),
        "TRAIN_MEAN_SHAPE": list(train_mean.shape),
        "TRAIN_STD_SHAPE": list(train_std.shape),
    }
    save_json(run_dir / "run_config.json", run_config)

    function_set = [
        "add",
        "sub",
        "mul",
        PROTECTED_DIV,
        "sin",
        "cos",
        "abs",
        "neg",
        PROTECTED_LOG,
        PROTECTED_SQRT,
        PROTECTED_INV,
    ]

    print("Training GP model...")
    gp = SymbolicRegressor(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        tournament_size=TOURNAMENT_SIZE,
        stopping_criteria=STOPPING_CRITERIA,
        const_range=(-5.0, 5.0),
        init_depth=INIT_DEPTH,
        init_method=INIT_METHOD,
        function_set=function_set,
        metric=STABILIZED_FITNESS,
        parsimony_coefficient=PARSIMONY_COEFFICIENT,
        p_crossover=P_CROSSOVER,
        p_subtree_mutation=P_SUBTREE_MUTATION,
        p_hoist_mutation=P_HOIST_MUTATION,
        p_point_mutation=P_POINT_MUTATION,
        p_point_replace=P_POINT_REPLACE,
        max_samples=MAX_SAMPLES,
        feature_names=train_data.feature_names,
        warm_start=False,
        low_memory=False,
        n_jobs=N_JOBS,
        verbose=VERBOSE,
        random_state=RANDOM_SEED,
    )

    gp.fit(X_train_fit, y_train_fit)

    print("Predicting train / val...")
    y_train_pred_raw = gp.predict(train_data.X)
    y_val_pred_raw = gp.predict(val_data.X)

    calibration_params = None
    if USE_LINEAR_CALIBRATION:
        print("Fitting linear calibration on full train predictions...")
        a, b = fit_linear_calibration(train_data.y, y_train_pred_raw)
        y_train_pred = apply_linear_calibration(y_train_pred_raw, a, b)
        y_val_pred = apply_linear_calibration(y_val_pred_raw, a, b)
        calibration_params = {"a": a, "b": b}
    else:
        y_train_pred = y_train_pred_raw
        y_val_pred = y_val_pred_raw

    metrics_rows = [
        {
            "split": "train",
            "rmse": rmse(train_data.y, y_train_pred),
            "nrmse": nrmse(train_data.y, y_train_pred),
        },
        {
            "split": "val",
            "rmse": rmse(val_data.y, y_val_pred),
            "nrmse": nrmse(val_data.y, y_val_pred),
        },
    ]
    save_metrics_csv(run_dir / "metrics.csv", metrics_rows)

    print("Train RMSE :", metrics_rows[0]["rmse"])
    print("Train NRMSE:", metrics_rows[0]["nrmse"])
    print("Val RMSE   :", metrics_rows[1]["rmse"])
    print("Val NRMSE  :", metrics_rows[1]["nrmse"])

    print("Saving best program...")
    best_program_str = str(gp._program)
    (run_dir / "best_program.txt").write_text(best_program_str, encoding="utf-8")

    if calibration_params is not None:
        save_json(run_dir / "linear_calibration.json", calibration_params)

    print("Saving training history...")
    run_details_df = pd.DataFrame(gp.run_details_)
    run_details_df.to_csv(run_dir / "training_history.csv", index=False)
    plot_convergence(run_details_df, plots_dir / "convergence.png")

    if SAVE_TRAIN_PREDICTIONS:
        train_pred_df = create_prediction_table(train_data, y_train_pred)
        train_pred_df.to_csv(run_dir / "train_predictions.csv", index=False)

    val_pred_df = create_prediction_table(val_data, y_val_pred)
    val_pred_df.to_csv(run_dir / "val_predictions.csv", index=False)

    if SAVE_VAL_PATIENT_PLOTS:
        plot_val_patients(
            pred_df=val_pred_df,
            target_column=val_data.target_column,
            output_dir=plots_dir / "val_patients",
        )

    if SAVE_VAL_OVERVIEW_PLOT:
        plot_val_overview(
            pred_df=val_pred_df,
            target_column=val_data.target_column,
            output_path=plots_dir / "val_all_patients_overview.png",
        )

    if SAVE_SCATTER_PLOT:
        plot_true_vs_pred_scatter(
            y_true=val_data.y,
            y_pred=y_val_pred,
            target_column=val_data.target_column,
            output_path=plots_dir / "val_true_vs_pred_scatter.png",
        )

    if SAVE_RESIDUAL_HIST:
        plot_residual_histogram(
            y_true=val_data.y,
            y_pred=y_val_pred,
            output_path=plots_dir / "val_residual_histogram.png",
        )

    print(f"Done. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()