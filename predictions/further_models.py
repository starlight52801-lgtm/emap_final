"""EMAP regression pipeline for Heart Rate, GSR, and Arousal prediction.

This script trains and compares multiple regression models on the EMAP CSV
files. It was designed to address the common failure mode where models mostly
predict the global mean instead of the actual temporal behavior.

Main ideas:
- Predict three targets independently:
    - heartrate_mean
    - GSR_mean
    - LABEL_SR_Arousal
- Use several model families for comparison:
    - Dummy baseline
    - ElasticNet
    - Random Forest
    - Histogram Gradient Boosting
    - MLP Regressor
- Reduce participant-specific baseline effects via per-file standardization.
- Add simple temporal context from IRPleth and Respir.
- Avoid target leakage by excluding all target columns from the feature matrix.
- Save metrics, predictions, model summaries, and plots to disk.

Before running:
- Adapt the paths in the CONFIG section near the bottom of the file.
- Then simply run:
    python predictions/further_models.py
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)



TARGETS: List[str] = ["heartrate_mean", "GSR_mean", "LABEL_SR_Arousal"]
NON_EEG_SIGNALS: List[str] = ["IRPleth_mean", "Respir_mean"]
FILE_PATTERN = re.compile(r"Features_P(?P<participant>\d+)-T(?P<trial>\d+)\.csv$", re.IGNORECASE)


@dataclass
class PipelineConfig:
    """Configuration for the EMAP regression pipeline.

    Attributes:
        train_dir: Directory containing training CSV files.
        val_dir: Directory containing validation CSV files.
        output_dir: Directory where all outputs will be written.
        random_seed: Random seed for reproducibility.
        max_train_rows: Optional cap for sampled training rows after preprocessing.
        max_val_rows: Optional cap for sampled validation rows after preprocessing. Set this to None if full patient plots across all trials should remain intact.
        add_temporal_features: Whether to add lag/rolling features from non-EEG signals.
        participantwise_standardize: Whether to z-score features separately per file.
        log_transform_eeg: Whether to apply log1p to EEG features after clipping negatives.
        lag_steps: Lag steps for temporal context.
        rolling_windows: Rolling window sizes for temporal context.
        quantile_clip_low: Lower training quantile used for clipping.
        quantile_clip_high: Upper training quantile used for clipping.
    """

    train_dir: str
    val_dir: str
    output_dir: str
    random_seed: int = 42
    max_train_rows: Optional[int] = 120000
    max_val_rows: Optional[int] = None
    add_temporal_features: bool = True
    participantwise_standardize: bool = True
    log_transform_eeg: bool = True
    lag_steps: Tuple[int, ...] = (1, 3, 5)
    rolling_windows: Tuple[int, ...] = (5, 15)
    quantile_clip_low: float = 0.001
    quantile_clip_high: float = 0.999


def set_global_seed(seed: int) -> None:
    """Set global random seeds.

    Args:
        seed: Random seed used for Python and NumPy.
    """
    random.seed(seed)
    np.random.seed(seed)


def extract_file_metadata(file_path: Path) -> Tuple[str, str]:
    """Extract participant and trial identifiers from a CSV filename.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Tuple containing participant ID and trial ID.
    """
    match = FILE_PATTERN.search(file_path.name)
    if match is None:
        return "unknown", "unknown"
    return match.group("participant"), match.group("trial")


def list_csv_files(folder: Path) -> List[Path]:
    """Return all CSV files within a folder.

    Args:
        folder: Directory containing CSV files.

    Returns:
        Sorted list of CSV file paths.
    """
    return sorted(folder.glob("*.csv"))


def add_temporal_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features based on IRPleth and Respir signals.

    Args:
        df: Input dataframe for a single file.

    Returns:
        Dataframe with additional temporal features.
    """
    out = df.copy()

    for signal_name in NON_EEG_SIGNALS:
        if signal_name not in out.columns:
            continue

        # Build simple temporal context that can help the model follow changes.
        # These features are intentionally conservative and only use non-target
        # signals to avoid target leakage.
        signal = out[signal_name]

        for lag in CONFIG.lag_steps:
            out[f"{signal_name}_lag_{lag}"] = signal.shift(lag)

        for window in CONFIG.rolling_windows:
            rolling = signal.rolling(window=window, min_periods=1)
            out[f"{signal_name}_rollmean_{window}"] = rolling.mean()
            out[f"{signal_name}_rollstd_{window}"] = rolling.std().fillna(0.0)

        out[f"{signal_name}_diff_1"] = signal.diff().fillna(0.0)
        out[f"{signal_name}_diff_3"] = signal.diff(3).fillna(0.0)

    return out


def participantwise_zscore(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Apply z-score normalization separately for one file.

    Args:
        df: Input dataframe.
        feature_columns: Columns to normalize.

    Returns:
        Dataframe with normalized feature columns.
    """
    out = df.copy()
    feature_block = out.loc[:, feature_columns]

    # Compute mean/std per file so that strong participant-specific baselines
    # do not dominate the learning problem.
    means = feature_block.mean(axis=0)
    stds = feature_block.std(axis=0).replace(0.0, 1.0)

    out.loc[:, feature_columns] = (feature_block - means) / stds
    return out


def load_split(folder: Path, add_temporal_features_flag: bool) -> pd.DataFrame:
    """Load and combine all CSV files from one split.

    Args:
        folder: Folder containing the split files.
        add_temporal_features_flag: Whether temporal context should be added.

    Returns:
        Combined dataframe with metadata columns.
    """
    csv_files = list_csv_files(folder)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    frames: List[pd.DataFrame] = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        participant_id, trial_id = extract_file_metadata(csv_path)
        df["participant_id"] = participant_id
        df["trial_id"] = trial_id
        df["source_file"] = csv_path.name
        df["time_index"] = np.arange(len(df), dtype=np.int64)

        if add_temporal_features_flag:
            df = add_temporal_context(df)

        frames.append(df)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Determine the usable feature columns.

    Args:
        df: Combined dataframe.

    Returns:
        Sorted list of numeric feature columns excluding targets and metadata.
    """
    metadata_columns = {"participant_id", "trial_id", "source_file", "time_index"}
    candidate_columns = []

    for column in df.columns:
        if column in TARGETS or column in metadata_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            candidate_columns.append(column)

    return sorted(candidate_columns)


def clip_and_log_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: Sequence[str],
    log_transform_eeg_flag: bool,
    quantile_low: float,
    quantile_high: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clip extreme values based on train quantiles and optionally log-transform EEG features.

    Args:
        train_df: Training dataframe.
        val_df: Validation dataframe.
        feature_columns: Feature columns to transform.
        log_transform_eeg_flag: Whether EEG columns should be log-transformed.
        quantile_low: Lower clipping quantile.
        quantile_high: Upper clipping quantile.

    Returns:
        Tuple of transformed train and validation dataframes.
    """
    train_out = train_df.copy()
    val_out = val_df.copy()

    lower = train_out.loc[:, feature_columns].quantile(quantile_low)
    upper = train_out.loc[:, feature_columns].quantile(quantile_high)

    train_out.loc[:, feature_columns] = train_out.loc[:, feature_columns].clip(lower=lower, upper=upper, axis=1)
    val_out.loc[:, feature_columns] = val_out.loc[:, feature_columns].clip(lower=lower, upper=upper, axis=1)

    if log_transform_eeg_flag:
        eeg_columns = [col for col in feature_columns if col.startswith("EEG_")]
        if eeg_columns:
            # EEG powers are often strongly right-skewed. A log transform can
            # stabilize the scale and make the regression problem easier.
            train_out.loc[:, eeg_columns] = np.log1p(np.clip(train_out.loc[:, eeg_columns], a_min=0.0, a_max=None))
            val_out.loc[:, eeg_columns] = np.log1p(np.clip(val_out.loc[:, eeg_columns], a_min=0.0, a_max=None))

    return train_out, val_out


def maybe_standardize_by_file(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: Sequence[str],
    enabled: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply file-wise standardization if enabled.

    Args:
        train_df: Training dataframe.
        val_df: Validation dataframe.
        feature_columns: Feature columns to normalize.
        enabled: Whether normalization is enabled.

    Returns:
        Tuple of transformed train and validation dataframes.
    """
    if not enabled:
        return train_df, val_df

    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []

    for _, group in train_df.groupby("source_file", sort=False):
        train_parts.append(participantwise_zscore(group, feature_columns))

    for _, group in val_df.groupby("source_file", sort=False):
        val_parts.append(participantwise_zscore(group, feature_columns))

    train_out = pd.concat(train_parts, axis=0, ignore_index=True)
    val_out = pd.concat(val_parts, axis=0, ignore_index=True)
    return train_out, val_out


def maybe_sample_rows(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    """Sample rows if a row cap is configured.

    Args:
        df: Input dataframe.
        max_rows: Maximum number of rows or None for all rows.
        seed: Random seed.

    Returns:
        Dataframe with at most max_rows rows.
    """
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary containing RMSE, NRMSE, MAE, and R².
    """
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    target_range = float(np.max(y_true) - np.min(y_true))
    nrmse = float(rmse / target_range) if target_range > 0 else float("nan")
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "NRMSE": nrmse, "MAE": mae, "R2": r2}


def compute_target_sample_weights(df: pd.DataFrame, target_name: str) -> np.ndarray:
    """Compute sample weights that emphasize extremes and temporal changes.

    Args:
        df: Dataframe containing the target, source file, and time index.
        target_name: Target column name.

    Returns:
        One weight per row.
    """
    y = df[target_name].astype(float)
    y_centered = y - y.median()
    y_scale = float(np.nanmedian(np.abs(y_centered)))
    if not np.isfinite(y_scale) or y_scale <= 1e-8:
        y_scale = float(y.std()) if float(y.std()) > 1e-8 else 1.0

    z_component = np.abs(y_centered / y_scale).clip(0.0, 4.0)

    delta_component = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for _, group_idx in df.groupby("source_file", sort=False).groups.items():
        group = df.loc[group_idx].sort_values("time_index")
        deltas = group[target_name].astype(float).diff().abs().fillna(0.0)
        delta_component.loc[group.index] = deltas

    delta_scale = float(delta_component.median())
    if not np.isfinite(delta_scale) or delta_scale <= 1e-8:
        delta_scale = float(delta_component.mean()) if float(delta_component.mean()) > 1e-8 else 1.0

    delta_component = (delta_component / delta_scale).clip(0.0, 4.0)
    weights = 1.0 + 0.35 * z_component.to_numpy() + 0.65 * delta_component.to_numpy()
    return weights.astype(float)


def fit_model_with_optional_weights(model: object, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> object:
    """Fit a model while forwarding sample weights whenever supported.

    Args:
        model: Estimator or pipeline.
        X: Training features.
        y: Training targets.
        sample_weight: Sample weights.

    Returns:
        Fitted model.
    """
    if isinstance(model, DummyRegressor):
        model.fit(X, y, sample_weight=sample_weight)
        return model

    if isinstance(model, Pipeline):
        last_step_name = model.steps[-1][0]
        fit_kwargs = {f"{last_step_name}__sample_weight": sample_weight}
        try:
            model.fit(X, y, **fit_kwargs)
            return model
        except TypeError:
            model.fit(X, y)
            return model

    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    return model


def build_model_zoo(seed: int) -> Dict[str, object]:
    """Create the model collection used for comparison.

    Args:
        seed: Random seed.

    Returns:
        Dictionary mapping model names to estimators.
    """
    model_zoo: Dict[str, object] = {
        "DummyMean": DummyRegressor(strategy="mean"),
        "ElasticNet": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", ElasticNet(alpha=0.0005, l1_ratio=0.15, max_iter=8000, random_state=seed)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=24,
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=seed,
                )),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_depth=8,
                    max_iter=350,
                    min_samples_leaf=30,
                    l2_regularization=0.05,
                    random_state=seed,
                )),
            ]
        ),
        "MLP": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    early_stopping=True,
                    validation_fraction=0.1,
                    max_iter=250,
                    random_state=seed,
                )),
            ]
        ),
    }
    return model_zoo


def fit_and_evaluate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_name: str,
    output_root: Path,
    seed: int,
) -> pd.DataFrame:
    """Train all models for one target and evaluate them on validation data.

    Args:
        train_df: Training dataframe.
        val_df: Validation dataframe.
        feature_columns: Feature columns.
        target_name: Target variable name.
        output_root: Output root path.
        seed: Random seed.

    Returns:
        Dataframe containing the metrics of all models.
    """
    target_dir = output_root / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    train_target_df = train_df[train_df[target_name].notna()].copy().reset_index(drop=True)
    val_target_df = val_df[val_df[target_name].notna()].copy().reset_index(drop=True)

    if train_target_df.empty:
        raise ValueError(f"Training data for target '{target_name}' contains only NaN values.")
    if val_target_df.empty:
        raise ValueError(f"Validation data for target '{target_name}' contains only NaN values.")

    X_train = train_target_df.loc[:, feature_columns].copy()
    y_train = train_target_df[target_name].astype(float).copy()
    X_val = val_target_df.loc[:, feature_columns].copy()
    y_val = val_target_df[target_name].astype(float).copy()

    sample_weight = compute_target_sample_weights(train_target_df, target_name)

    results: List[Dict[str, float]] = []
    model_zoo = build_model_zoo(seed)

    for model_name, model in model_zoo.items():
        print(f"Training {target_name} -> {model_name}")
        fitted_model = clone(model) if hasattr(model, "get_params") else model
        fitted_model = fit_model_with_optional_weights(fitted_model, X_train, y_train, sample_weight)
        val_preds = np.asarray(fitted_model.predict(X_val), dtype=float)

        metrics = compute_metrics(y_val.to_numpy(), val_preds)
        result_row: Dict[str, float] = {"target": target_name, "model": model_name, **metrics}
        results.append(result_row)

        prediction_df = val_target_df.loc[:, ["source_file", "participant_id", "trial_id", "time_index", target_name]].copy()
        prediction_df.rename(columns={target_name: "y_true"}, inplace=True)
        prediction_df["y_pred"] = val_preds
        prediction_df["abs_error"] = np.abs(prediction_df["y_true"] - prediction_df["y_pred"])
        prediction_df.to_csv(target_dir / f"predictions_{model_name}.csv", index=False)

        save_patient_prediction_plots(
            prediction_df=prediction_df,
            target_name=target_name,
            model_name=model_name,
            output_dir=target_dir / f"plots_{model_name}",
        )

    metrics_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    metrics_df.to_csv(target_dir / "metrics.csv", index=False)
    return metrics_df


def save_patient_prediction_plots(
    prediction_df: pd.DataFrame,
    target_name: str,
    model_name: str,
    output_dir: Path,
) -> None:
    """Save one plot per validation patient across all trials.

    Each patient plot concatenates all available validation trials in ascending
    trial order. Trial boundaries are marked with dashed vertical lines so the
    full patient trajectory remains visible while still showing where one trial
    ends and the next begins.

    Args:
        prediction_df: Validation predictions dataframe.
        target_name: Name of the predicted target.
        model_name: Model name for the plot title.
        output_dir: Directory where patient plots will be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_ids = sorted(prediction_df["participant_id"].dropna().astype(str).unique().tolist())
    if not participant_ids:
        return

    for participant_id in participant_ids:
        participant_df = prediction_df[prediction_df["participant_id"].astype(str) == participant_id].copy()
        if participant_df.empty:
            continue

        participant_df["trial_id_numeric"] = pd.to_numeric(participant_df["trial_id"], errors="coerce")
        participant_df = participant_df.sort_values(
            by=["trial_id_numeric", "trial_id", "time_index", "source_file"],
            kind="mergesort",
        ).reset_index(drop=True)

        trial_blocks = []
        boundary_positions: List[int] = []
        tick_positions: List[float] = []
        tick_labels: List[str] = []
        running_offset = 0

        grouped_trials = participant_df.groupby(["trial_id", "source_file"], sort=False)
        for index, ((trial_id, source_file), trial_df) in enumerate(grouped_trials):
            trial_df = trial_df.sort_values("time_index").copy().reset_index(drop=True)
            trial_df["global_index"] = np.arange(len(trial_df), dtype=np.int64) + running_offset
            trial_blocks.append(trial_df)

            tick_positions.append(float(trial_df["global_index"].iloc[0] + trial_df["global_index"].iloc[-1]) / 2.0)
            tick_labels.append(f"T{trial_id}")

            running_offset += len(trial_df)
            if index < len(grouped_trials) - 1:
                boundary_positions.append(running_offset - 0.5)

        if not trial_blocks:
            continue

        plot_df = pd.concat(trial_blocks, axis=0, ignore_index=True)

        plt.figure(figsize=(18, 6))
        plt.plot(
            plot_df["global_index"].to_numpy(),
            plot_df["y_true"].to_numpy(),
            label="Ground truth",
            linewidth=1.5,
        )
        plt.plot(
            plot_df["global_index"].to_numpy(),
            plot_df["y_pred"].to_numpy(),
            label="Prediction",
            linewidth=1.2,
            alpha=0.9,
        )

        for boundary in boundary_positions:
            plt.axvline(boundary, linestyle="--", linewidth=0.9, alpha=0.8)

        if tick_positions:
            plt.xticks(tick_positions, tick_labels, rotation=0)

        plt.title(f"{target_name} - {model_name} - Participant {participant_id} (all validation trials)")
        plt.xlabel("Concatenated time index across trials")
        plt.ylabel(target_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"patient_{participant_id}_{model_name}.png", dpi=180)
        plt.close()


def save_summary_report(
    config: PipelineConfig,
    output_root: Path,
    feature_columns: Sequence[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    all_metrics: pd.DataFrame,
) -> None:
    """Write a short textual summary to disk.

    Args:
        config: Pipeline configuration.
        output_root: Output root path.
        feature_columns: Feature columns used.
        train_df: Training dataframe.
        val_df: Validation dataframe.
        all_metrics: Combined metrics dataframe.
    """
    lines: List[str] = []
    lines.append("EMAP Multitarget Regression Pipeline Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Configuration:")
    lines.append(json.dumps(asdict(config), indent=2))
    lines.append("")
    lines.append(f"Train rows: {len(train_df)}")
    lines.append(f"Validation rows: {len(val_df)}")
    lines.append(f"Train participants: {train_df['participant_id'].nunique()}")
    lines.append(f"Validation participants: {val_df['participant_id'].nunique()}")
    lines.append(f"Number of features used: {len(feature_columns)}")
    lines.append("")
    lines.append("Best model per target (sorted by RMSE):")

    for target_name in TARGETS:
        target_rows = all_metrics[all_metrics["target"] == target_name].sort_values("RMSE", ascending=True)
        if target_rows.empty:
            continue
        best = target_rows.iloc[0]
        lines.append(
            f"- {target_name}: {best['model']} | "
            f"RMSE={best['RMSE']:.4f}, NRMSE={best['NRMSE']:.4f}, "
            f"MAE={best['MAE']:.4f}, R2={best['R2']:.4f}"
        )

    (output_root / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the full EMAP regression experiment."""
    set_global_seed(CONFIG.random_seed)

    train_dir = Path(CONFIG.train_dir)
    val_dir = Path(CONFIG.val_dir)
    output_root = Path(CONFIG.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading training files...")
    train_df = load_split(train_dir, add_temporal_features_flag=CONFIG.add_temporal_features)

    print("Loading validation files...")
    val_df = load_split(val_dir, add_temporal_features_flag=CONFIG.add_temporal_features)

    # Determine a common numeric feature space for both splits.
    train_features = set(get_feature_columns(train_df))
    val_features = set(get_feature_columns(val_df))
    feature_columns = sorted(train_features.intersection(val_features))

    print(f"Common feature count before transforms: {len(feature_columns)}")

    train_df, val_df = clip_and_log_transform(
        train_df=train_df,
        val_df=val_df,
        feature_columns=feature_columns,
        log_transform_eeg_flag=CONFIG.log_transform_eeg,
        quantile_low=CONFIG.quantile_clip_low,
        quantile_high=CONFIG.quantile_clip_high,
    )

    train_df, val_df = maybe_standardize_by_file(
        train_df=train_df,
        val_df=val_df,
        feature_columns=feature_columns,
        enabled=CONFIG.participantwise_standardize,
    )

    train_df = maybe_sample_rows(train_df, CONFIG.max_train_rows, CONFIG.random_seed)
    val_df = maybe_sample_rows(val_df, CONFIG.max_val_rows, CONFIG.random_seed)

    print(
        f"Prepared data -> train rows: {len(train_df)}, val rows: {len(val_df)}, "
        f"train participants: {train_df['participant_id'].nunique()}, "
        f"val participants: {val_df['participant_id'].nunique()}"
    )

    all_metrics_list: List[pd.DataFrame] = []

    for target_name in TARGETS:
        metrics_df = fit_and_evaluate_models(
            train_df=train_df,
            val_df=val_df,
            feature_columns=feature_columns,
            target_name=target_name,
            output_root=output_root,
            seed=CONFIG.random_seed,
        )
        all_metrics_list.append(metrics_df)

    all_metrics = pd.concat(all_metrics_list, axis=0, ignore_index=True)
    all_metrics.to_csv(output_root / "all_metrics.csv", index=False)

    save_summary_report(
        config=CONFIG,
        output_root=output_root,
        feature_columns=feature_columns,
        train_df=train_df,
        val_df=val_df,
        all_metrics=all_metrics,
    )

    print("Done.")
    print(f"Results written to: {output_root}")


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = PipelineConfig(
    train_dir=r"data/train",
    val_dir=r"data/val",
    output_dir=r"predictions/output/further_models",
    random_seed=42,
    max_train_rows=120000,
    max_val_rows=80000,
    add_temporal_features=True,
    participantwise_standardize=True,
    log_transform_eeg=True,
    lag_steps=(1, 3, 5),
    rolling_windows=(5, 15),
    quantile_clip_low=0.001,
    quantile_clip_high=0.999,
)


if __name__ == "__main__":
    main()
