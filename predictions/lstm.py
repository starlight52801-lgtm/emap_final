"""PyTorch LSTM regression for EMAP targets with validation patient plots.

This script is a PyTorch adaptation of the original TensorFlow/Keras version.
It keeps the overall logic close to the original workflow while adding:

- training on participant-based folds built from files in ``data/train``
- validation plots on files from ``data/val``
- predictions on ``data/val``
- RMSE / NRMSE on the validation split for comparison with other algorithms

Targets:
    - heartrate_mean
    - GSR_mean
    - LABEL_SR_Arousal

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
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# User configuration
# =============================================================================

# Select target here:
# "heartrate_mean", "GSR_mean", or "LABEL_SR_Arousal"
TARGET_COLUMN = "LABEL_SR_Arousal"

# Training files used for fold creation
TRAIN_PATH = "data/train"

# External validation files used for final comparison plots / metrics
VAL_PATH = "data/val"

# Output root
OUTPUT_ROOT = "predictions/output/lstm"

# Training parameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Cross-validation
MAX_FOLDS = 5

# Device configuration
CUDA_DEVICE = "0"

# Reproducibility
RANDOM_SEED = 42


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class PreparedSplit:
    """Container for one prepared data split.

    Attributes:
        X: Input features as NumPy array.
        y: Target values as NumPy array.
        scaler_X: Fitted feature scaler.
        scaler_y: Fitted target scaler.
    """

    X: np.ndarray
    y: np.ndarray
    scaler_X: Optional[StandardScaler]
    scaler_y: Optional[StandardScaler]


@dataclass
class PreparedRows:
    """Prepared rows with metadata for validation plotting.

    Attributes:
        X: Input features in 2D shape before LSTM reshaping.
        y: Target values in original scale.
        meta: Row-level metadata aligned with X and y.
    """

    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


# =============================================================================
# Utility functions
# =============================================================================


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_device(cuda_device: str) -> torch.device:
    """Get the computation device.

    Args:
        cuda_device: CUDA device index as string.

    Returns:
        A PyTorch device object.
    """
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        return torch.device("cuda")
    return torch.device("cpu")



def nrmse(rmse_value: float, y_true: np.ndarray) -> float:
    """Compute normalized root mean squared error.

    Args:
        rmse_value: Root mean squared error.
        y_true: Ground-truth target values in original scale.

    Returns:
        Normalized RMSE. Returns NaN if the target range is zero.
    """
    y_true = np.asarray(y_true).reshape(-1)
    target_range = float(np.max(y_true) - np.min(y_true))

    if target_range == 0.0:
        return float("nan")

    return float(rmse_value / target_range)



def append_list_as_row(file_name: str, list_of_elem: List[str]) -> None:
    """Append one row to a CSV file.

    Args:
        file_name: CSV file path.
        list_of_elem: Row values to append.
    """
    with open(file_name, "a+", newline="") as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)



def parse_participant_id(filename: str) -> Optional[str]:
    """Extract participant ID from filename.

    Args:
        filename: Input CSV filename.

    Returns:
        The participant ID as string if found, otherwise None.
    """
    basename = os.path.basename(filename)
    match = re.search(r"P(\d{3})", basename)
    if match is None:
        return None
    return match.group(1)



def parse_trial_id(filename: str) -> Optional[int]:
    """Extract trial ID from filename.

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



def load_all_files(path: str) -> List[str]:
    """Load all CSV file paths from a directory.

    Args:
        path: Folder path.

    Returns:
        Sorted list of CSV paths.

    Raises:
        FileNotFoundError: If no CSV files are found.
    """
    all_files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    return all_files



def build_participant_file_mapping(all_files: List[str]) -> Dict[str, List[pd.DataFrame]]:
    """Group CSV files by participant.

    Args:
        all_files: List of CSV file paths.

    Returns:
        Dictionary mapping participant ID to a list of trial DataFrames.
    """
    participant_to_files: Dict[str, List[pd.DataFrame]] = {}

    for filename in all_files:
        participant_id = parse_participant_id(filename)
        if participant_id is None:
            continue

        df = pd.read_csv(filename, encoding="ISO-8859-1", header=0)

        if participant_id not in participant_to_files:
            participant_to_files[participant_id] = []

        participant_to_files[participant_id].append(df)

    return participant_to_files



def create_fold_lists(
    participant_to_files: Dict[str, List[pd.DataFrame]],
    max_folds: int,
) -> List[List[pd.DataFrame]]:
    """Create participant-based folds.

    Args:
        participant_to_files: Mapping from participant ID to list of trial DataFrames.
        max_folds: Maximum number of folds.

    Returns:
        List of folds, each containing a flat list of DataFrames.
    """
    participant_ids = sorted(participant_to_files.keys())
    participant_groups = [participant_to_files[participant_id] for participant_id in participant_ids]

    n_folds = min(max_folds, len(participant_groups))
    split_groups = [participant_groups[i::n_folds] for i in range(n_folds)]

    list_of_lists: List[List[pd.DataFrame]] = []
    for fold_idx, fold in enumerate(split_groups):
        flat_fold: List[pd.DataFrame] = []

        for participant_file_list in fold:
            flat_fold.extend(participant_file_list)

        list_of_lists.append(flat_fold)
        print(f"Fold {fold_idx}: {len(flat_fold)} files")

    return list_of_lists



def clean_target_rows(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows with invalid target values.

    Args:
        X_df: Feature DataFrame.
        y_df: Target DataFrame.
        target_column: Target column name.

    Returns:
        Tuple of cleaned (X_df, y_df).
    """
    valid_mask = np.isfinite(y_df[target_column].to_numpy())
    X_df = X_df.loc[valid_mask].reset_index(drop=True)
    y_df = y_df.loc[valid_mask].reset_index(drop=True)
    return X_df, y_df



def prepare_xy_train(
    data: pd.DataFrame,
    target_column: str,
) -> PreparedSplit:
    """Prepare train split and fit scalers.

    Args:
        data: Input training DataFrame.
        target_column: Target column name.

    Returns:
        Prepared training split with fitted scalers.

    Raises:
        KeyError: If the target column is missing.
    """
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in data.")

    data = data.fillna(0).copy()

    X_df = data.loc[:, data.columns != target_column].copy()
    y_df = data.loc[:, data.columns == target_column].copy()

    X_df, y_df = clean_target_rows(X_df, y_df, target_column)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X_df)
    y = scaler_y.fit_transform(y_df.values)

    return PreparedSplit(X=X, y=y, scaler_X=scaler_X, scaler_y=scaler_y)



def prepare_xy_apply(
    data: pd.DataFrame,
    target_column: str,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
) -> PreparedSplit:
    """Prepare validation or test split using train-fitted scalers.

    Args:
        data: Input DataFrame.
        target_column: Target column name.
        scaler_X: Feature scaler fitted on train.
        scaler_y: Target scaler fitted on train.

    Returns:
        Prepared split with transformed X and y.

    Raises:
        KeyError: If the target column is missing.
    """
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in data.")

    data = data.fillna(0).copy()

    X_df = data.loc[:, data.columns != target_column].copy()
    y_df = data.loc[:, data.columns == target_column].copy()

    X_df, y_df = clean_target_rows(X_df, y_df, target_column)

    X = scaler_X.transform(X_df)
    y = scaler_y.transform(y_df.values)

    return PreparedSplit(X=X, y=y, scaler_X=None, scaler_y=None)



def prepare_val_rows(
    val_files: List[str],
    target_column: str,
    scaler_X: StandardScaler,
) -> PreparedRows:
    """Prepare row-level validation data with metadata.

    Args:
        val_files: Validation CSV file paths.
        target_column: Target column name.
        scaler_X: Feature scaler fitted on train.

    Returns:
        Prepared rows with scaled features, original targets, and metadata.
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
        raise ValueError("No usable validation rows were found.")

    X = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    meta = pd.concat(meta_parts, axis=0, ignore_index=True)

    return PreparedRows(X=X, y=y, meta=meta)



def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    """Reshape 2D input to 3D LSTM input.

    Args:
        X: Input array of shape (n_samples, n_features).

    Returns:
        Reshaped array of shape (n_samples, n_features, 1).
    """
    return np.reshape(X, (X.shape[0], X.shape[1], 1)).astype(np.float32)



def inverse_transform_targets(
    y_scaled: np.ndarray,
    scaler_y: StandardScaler,
) -> np.ndarray:
    """Transform scaled target values back to original scale.

    Args:
        y_scaled: Scaled target values.
        scaler_y: Target scaler fitted on train.

    Returns:
        Target values in original scale.
    """
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    return scaler_y.inverse_transform(y_scaled).reshape(-1)


# =============================================================================
# Model definition
# =============================================================================


class LSTMRegressor(nn.Module):
    """LSTM regressor matching the original architecture idea."""

    def __init__(self, hidden_size: int = 150, dropout: float = 0.2) -> None:
        """Initialize the LSTM regressor.

        Args:
            hidden_size: Number of hidden units.
            dropout: Dropout probability.
        """
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dense_time = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout4 = nn.Dropout(dropout)

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, 1).

        Returns:
            Predicted tensor of shape (batch, 1).
        """
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x = self.dense_time(x)
        x = self.dropout3(x)

        x, _ = self.lstm3(x)
        x = self.dropout4(x)

        x = x[:, -1, :]
        x = self.output_layer(x)
        return x


# =============================================================================
# Training / evaluation
# =============================================================================


def create_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader.

    Args:
        X: Input array.
        y: Target array.
        batch_size: Batch size.
        shuffle: Whether to shuffle the dataset.

    Returns:
        A PyTorch DataLoader.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch.

    Args:
        model: PyTorch model.
        loader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Computation device.

    Returns:
        Mean training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model loss on a dataset.

    Args:
        model: PyTorch model.
        loader: Data loader.
        criterion: Loss function.
        device: Computation device.

    Returns:
        Mean loss.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run model prediction.

    Args:
        model: Trained PyTorch model.
        X: Input array.
        device: Computation device.
        batch_size: Prediction batch size.

    Returns:
        Predicted values as NumPy array.
    """
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds_list: List[np.ndarray] = []

    for (X_batch,) in loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        preds_list.append(preds.cpu().numpy())

    return np.concatenate(preds_list, axis=0).reshape(-1, 1)


# =============================================================================
# Plot helpers
# =============================================================================


def save_patient_validation_plots(
    pred_df: pd.DataFrame,
    target_name: str,
    output_dir: str,
) -> None:
    """Save one validation plot per patient with all trials concatenated.

    Args:
        pred_df: Prediction DataFrame containing metadata, y_true, and y_pred.
        target_name: Human-readable target name.
        output_dir: Output directory for plot files.
    """
    os.makedirs(output_dir, exist_ok=True)

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
        plt.plot(y_pred, label="Predicted")

        for boundary in boundary_positions[:-1]:
            plt.axvline(boundary, linestyle="--", linewidth=1.0, color="gray")

        plt.title(f"Validation - Patient {patient_id} - {target_name}")
        plt.xlabel("Concatenated time steps across trials")
        plt.ylabel(target_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"patient_{patient_id}_{target_name}.png"), dpi=150)
        plt.close()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the full PyTorch LSTM workflow with validation plotting."""
    set_seed(RANDOM_SEED)
    device = get_device(CUDA_DEVICE)

    print(f"Using device: {device}")

    target_to_name = {
        "heartrate_mean": "heart_rate",
        "GSR_mean": "gsr",
        "LABEL_SR_Arousal": "arousal",
    }

    if TARGET_COLUMN not in target_to_name:
        raise ValueError(
            "TARGET_COLUMN must be one of: "
            "'heartrate_mean', 'GSR_mean', 'LABEL_SR_Arousal'"
        )

    target_name = target_to_name[TARGET_COLUMN]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        OUTPUT_ROOT,
        f"output_abgabe_{target_name}",
        f"run_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val_patient_plots"), exist_ok=True)

    print("Results will be saved to:", output_dir)

    results_csv = os.path.join(output_dir, f"{target_name}_results.csv")
    val_pred_csv = os.path.join(output_dir, f"{target_name}_val_predictions_all_folds.csv")

    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="") as f:
            header = ["Fold", "Split", "RMSE", "NRMSE"]
            filewriter = csv.DictWriter(f, fieldnames=header)
            filewriter.writeheader()

    print("Reading training data...")
    train_files = load_all_files(TRAIN_PATH)

    print("Reading external validation data...")
    val_files = load_all_files(VAL_PATH)

    li = []
    for filename in train_files:
        df = pd.read_csv(filename, encoding="ISO-8859-1", header=0)
        li.append(df)

    dataset = pd.concat(li, axis=0, ignore_index=True).fillna(0)

    X_overview = dataset.loc[:, dataset.columns != TARGET_COLUMN]
    y_overview = dataset.loc[:, dataset.columns == TARGET_COLUMN].values

    print("Train X is :", X_overview.shape)
    print("Train y is :", y_overview.shape)

    participant_to_files = build_participant_file_mapping(train_files)
    participant_ids = sorted(participant_to_files.keys())

    print("Number of training participants with data:", len(participant_ids))

    list_of_lists = create_fold_lists(
        participant_to_files=participant_to_files,
        max_folds=MAX_FOLDS,
    )

    all_val_pred_dfs: List[pd.DataFrame] = []

    for i in range(len(list_of_lists)):
        print("*********** Splitting test and train...")
        train_lists = list_of_lists[0:i] + list_of_lists[i + 1 :]
        test_list = list_of_lists[i]

        print("Train groups:", len(train_lists))
        print("Test files:", len(test_list))

        print("*********** Splitting train and eval...")
        second_list_of_lists = train_lists

        for j in range(0, 1):
            if i < len(list_of_lists) - 1:
                train_lists = second_list_of_lists[0:j] + second_list_of_lists[j + 1 :]
                val_list = second_list_of_lists[i]
            else:
                train_lists = second_list_of_lists[0:j] + second_list_of_lists[j + 1 :]
                val_list = second_list_of_lists[j]

        train = list(itertools.chain(*train_lists))

        train_data = pd.concat(train, axis=0, ignore_index=True).fillna(0)
        test_data = pd.concat(test_list, axis=0, ignore_index=True).fillna(0)
        inner_val_data = pd.concat(val_list, axis=0, ignore_index=True).fillna(0)

        train_prepared = prepare_xy_train(train_data, TARGET_COLUMN)

        test_prepared = prepare_xy_apply(
            test_data,
            TARGET_COLUMN,
            train_prepared.scaler_X,
            train_prepared.scaler_y,
        )
        inner_val_prepared = prepare_xy_apply(
            inner_val_data,
            TARGET_COLUMN,
            train_prepared.scaler_X,
            train_prepared.scaler_y,
        )

        # External validation set from data/val
        val_rows = prepare_val_rows(
            val_files=val_files,
            target_column=TARGET_COLUMN,
            scaler_X=train_prepared.scaler_X,
        )

        X_train = reshape_for_lstm(train_prepared.X)
        y_train = train_prepared.y.astype(np.float32)

        X_test = reshape_for_lstm(test_prepared.X)
        y_test = test_prepared.y.astype(np.float32)

        X_inner_val = reshape_for_lstm(inner_val_prepared.X)
        y_inner_val = inner_val_prepared.y.astype(np.float32)

        X_val_external = reshape_for_lstm(val_rows.X)

        print("Shapes of X_train, X_test, X_inner_val, X_val_external...")
        print("X_train       :", X_train.shape)
        print("X_test        :", X_test.shape)
        print("X_inner_val   :", X_inner_val.shape)
        print("X_val_external:", X_val_external.shape)

        train_loader = create_data_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
        inner_val_loader = create_data_loader(X_inner_val, y_inner_val, BATCH_SIZE, shuffle=False)

        model = LSTMRegressor(hidden_size=150, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        print("Training...")
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            inner_val_loss = evaluate_loss(
                model=model,
                loader=inner_val_loader,
                criterion=criterion,
                device=device,
            )
            print(
                f"Fold {i} | Epoch {epoch + 1}/{EPOCHS} | "
                f"Train loss: {train_loss:.6f} | Inner val loss: {inner_val_loss:.6f}"
            )

        # Fold test predictions
        y_pred_test_scaled = predict(model, X_test, device=device)
        y_test_scaled = y_test.reshape(-1, 1)

        y_pred_test = inverse_transform_targets(y_pred_test_scaled, train_prepared.scaler_y)
        y_test_original = inverse_transform_targets(y_test_scaled, train_prepared.scaler_y)

        mse_test = mean_squared_error(y_test_original, y_pred_test)
        rmse_test = sqrt(mse_test)
        nrmse_test = nrmse(rmse_test, y_test_original)

        print("Results on Fold:", i)
        print("Test RMSE:", rmse_test)
        print("Test NRMSE:", nrmse_test)

        append_list_as_row(results_csv, [str(i), "fold_test", str(rmse_test), str(nrmse_test)])

        # External validation predictions on data/val
        y_pred_val_scaled = predict(model, X_val_external, device=device)
        y_pred_val = inverse_transform_targets(y_pred_val_scaled, train_prepared.scaler_y)

        mse_val = mean_squared_error(val_rows.y, y_pred_val)
        rmse_val = sqrt(mse_val)
        nrmse_val = nrmse(rmse_val, val_rows.y)

        print("Validation RMSE (data/val):", rmse_val)
        print("Validation NRMSE (data/val):", nrmse_val)

        append_list_as_row(results_csv, [str(i), "external_val", str(rmse_val), str(nrmse_val)])

        pred_df = val_rows.meta.copy()
        pred_df["y_true"] = val_rows.y
        pred_df["y_pred"] = y_pred_val
        pred_df["fold"] = i
        all_val_pred_dfs.append(pred_df)

        fold_plot_dir = os.path.join(output_dir, "val_patient_plots", f"fold_{i}")
        save_patient_validation_plots(
            pred_df=pred_df,
            target_name=target_name,
            output_dir=fold_plot_dir,
        )

    if all_val_pred_dfs:
        full_val_pred_df = pd.concat(all_val_pred_dfs, axis=0, ignore_index=True)
        full_val_pred_df.to_csv(val_pred_csv, index=False)

        summary_rows = []
        for fold_id, fold_df in full_val_pred_df.groupby("fold", sort=True):
            fold_rmse = sqrt(mean_squared_error(fold_df["y_true"], fold_df["y_pred"]))
            fold_nrmse = nrmse(fold_rmse, fold_df["y_true"].to_numpy())
            summary_rows.append({
                "fold": int(fold_id),
                "rmse_external_val": fold_rmse,
                "nrmse_external_val": fold_nrmse,
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(output_dir, f"{target_name}_external_val_summary.csv"), index=False)

    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
