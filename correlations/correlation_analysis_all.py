"""
Correlation analysis of all training and validation data.

Main functionality:
- Load train and validation CSV files from paths defined in the code.
- Compute feature-feature correlations for train and validation.
- Create full correlation heatmaps.

Usage:
    python correlations/correlation_analysis_all.py

Before running, edit the configuration block in `main()` and set the paths that
match your machine.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class AnalysisConfig:
    """Configuration container for the EDA pipeline.

    Attributes:
        train_input: Path to a training CSV file or a directory containing CSVs.
        val_input: Path to a validation CSV file or a directory containing CSVs.
        output_dir: Directory where all plots and tables will be written.
        max_rows_per_split: Optional maximum number of rows per split after
            loading. Use ``None`` to keep all rows.
        random_state: Random seed for reproducible row subsampling.
        participant_regex: Regex used to extract participant IDs from filenames.
        target_candidates: Columns for which target-correlation tables should be
            created if present.
        histogram_feature_limit: Maximum number of histograms to create.
        top_corr_pairs: Number of strongest feature-feature pairs to export.
        min_rows_for_participant_corr: Minimum number of rows required for a
            participant to be included in participant-wise correlation analysis.
    """

    train_input: str
    val_input: str
    output_dir: str
    max_rows_per_split: Optional[int] = 50000
    random_state: int = 42
    participant_regex: str = r"P(\d{3})"
    target_candidates: Tuple[str, ...] = (
        "LABEL_SR_Arousal",
        "heartrate_mean",
        "GSR_mean",
        "Respir_mean",
        "IRPleth_mean",
    )
    histogram_feature_limit: int = 12
    top_corr_pairs: int = 100
    min_rows_for_participant_corr: int = 30


def collect_csv_files(input_path: str) -> List[Path]:
    """Collect CSV files from a file or directory path.

    Args:
        input_path: File path to a CSV or directory path containing CSV files.

    Returns:
        A sorted list of CSV paths.

    Raises:
        FileNotFoundError: If the path does not exist or no CSV files are found.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise FileNotFoundError(f"Expected a CSV file, got: {input_path}")
        return [path]

    files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in directory: {input_path}")
    return files



def extract_participant_id(file_path: Path, participant_regex: str) -> str:
    """Extract a participant identifier from a CSV filename.

    Args:
        file_path: Path to the input CSV file.
        participant_regex: Regex used to detect participant IDs.

    Returns:
        Extracted participant ID if available, otherwise the stem of the file.
    """
    match = re.search(participant_regex, file_path.name)
    if match:
        return match.group(1)
    return file_path.stem



def load_split_dataframe(
    input_path: str,
    split_name: str,
    participant_regex: str,
    max_rows: Optional[int],
    random_state: int,
) -> pd.DataFrame:
    """Load one split from CSV files and append metadata columns.

    Args:
        input_path: File or directory path for the split.
        split_name: Human-readable split name, e.g. ``train`` or ``val``.
        participant_regex: Regex used to derive participant IDs from filenames.
        max_rows: Optional row cap applied after concatenation.
        random_state: Seed for reproducible subsampling.

    Returns:
        Concatenated DataFrame with ``__split__``, ``__source_file__``, and
        ``__participant_id__`` helper columns.
    """
    files = collect_csv_files(input_path)
    frames: List[pd.DataFrame] = []

    for file_path in files:
        df = pd.read_csv(file_path)
        df["__split__"] = split_name
        df["__source_file__"] = file_path.name
        df["__participant_id__"] = extract_participant_id(file_path, participant_regex)
        frames.append(df)

    combined = pd.concat(frames, axis=0, ignore_index=True)

    # Optional downsampling keeps the script practical for very large datasets.
    if max_rows is not None and len(combined) > max_rows:
        combined = combined.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    return combined



def get_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns excluding helper metadata columns.

    Args:
        df: Input DataFrame.

    Returns:
        List of numeric feature column names.
    """
    excluded_prefix = "__"
    columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if not col.startswith(excluded_prefix):
            columns.append(col)
    return columns



def prepare_output_dirs(output_dir: str) -> Tuple[Path, Path, Path]:
    """Create output directory structure.

    Args:
        output_dir: Root output directory.

    Returns:
        Tuple of paths for root, plot, and table directories.
    """
    root = Path(output_dir)
    plots_dir = root / "plots"
    tables_dir = root / "tables"
    text_dir = root / "text"

    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    return root, plots_dir, tables_dir



def compute_correlation_matrix(df: pd.DataFrame, columns: Sequence[str], method: str) -> pd.DataFrame:
    """Compute a correlation matrix for selected columns.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to include.
        method: Correlation method, e.g. ``pearson`` or ``spearman``.

    Returns:
        Correlation matrix as a DataFrame.
    """
    return df.loc[:, columns].corr(method=method)



def _compute_cluster_order(corr_df: pd.DataFrame) -> List[int]:
    """Compute a robust clustering order for a correlation matrix.

    The function converts correlations to distances via ``1 - |corr|`` and then
    guards against numerical issues that can otherwise produce tiny negative
    distances and crash SciPy/Seaborn clustering.

    Args:
        corr_df: Square correlation-like matrix.

    Returns:
        Reordered feature indices. If clustering fails, the original order is
        returned.
    """
    values = corr_df.to_numpy(dtype=float, copy=True)

    # Replace invalid entries so the clustering stays stable even if some
    # participant-wise correlations contain NaNs.
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

    # Correlations should be in [-1, 1], but tiny floating-point overshoots can
    # happen after averaging or subtraction.
    values = np.clip(values, -1.0, 1.0)

    # Distances for clustering are based on the absolute correlation strength.
    distance = 1.0 - np.abs(values)

    # Numerical clean-up: force symmetry, zero diagonal, and non-negative values.
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)

    try:
        condensed = squareform(distance, checks=True)
        linkage_matrix = linkage(condensed, method="average")
        return leaves_list(linkage_matrix).tolist()
    except Exception:
        return list(range(len(corr_df)))



def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    output_path: Path,
    title: str,
    clustered: bool = False,
    annotate: bool = False,
    vmin: float = -1.0,
    vmax: float = 1.0,
    center: float = 0.0,
) -> None:
    """Plot and save a correlation heatmap.

    Args:
        corr_df: Correlation matrix.
        output_path: Destination image path.
        title: Plot title.
        clustered: Whether to reorder features using hierarchical clustering.
        annotate: Whether to annotate cells with values. Usually only suitable
            for very small matrices.
        vmin: Lower color scale bound.
        vmax: Upper color scale bound.
        center: Center value of the colormap.
    """
    plot_df = corr_df.copy()

    # Replace invalid values so plotting and clustering do not crash.
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if clustered and len(plot_df) > 2:
        order = _compute_cluster_order(plot_df)
        plot_df = plot_df.iloc[order, order]

    # Dynamic sizing keeps the plot readable even with many features.
    size = max(12, min(40, int(len(plot_df) * 0.18)))
    fig, ax = plt.subplots(figsize=(size, size), dpi=200)
    sns.heatmap(
        plot_df,
        ax=ax,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=True,
        xticklabels=True,
        yticklabels=True,
        annot=annotate,
        annot_kws={"size": 5},
        cbar_kws={"shrink": 0.75, "label": "Correlation"},
    )
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", labelrotation=90, labelsize=5)
    ax.tick_params(axis="y", labelsize=5)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def compute_top_correlation_pairs(
    corr_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Return the strongest feature-feature correlation pairs.

    Args:
        corr_df: Symmetric feature-feature correlation matrix.
        top_n: Number of top pairs to return.

    Returns:
        Long-form DataFrame with the strongest absolute correlations.
    """
    upper_mask = np.triu(np.ones(corr_df.shape, dtype=bool), k=1)
    stacked = corr_df.where(upper_mask).stack().reset_index()
    stacked.columns = ["feature_a", "feature_b", "correlation"]
    stacked["abs_correlation"] = stacked["correlation"].abs()
    stacked = stacked.sort_values("abs_correlation", ascending=False)
    return stacked.head(top_n).reset_index(drop=True)



def compute_target_correlations(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    target_candidates: Sequence[str],
) -> pd.DataFrame:
    """Compute feature-target correlations for all available targets.

    Args:
        df: Input DataFrame.
        numeric_columns: Numeric columns that can be correlated.
        target_candidates: Candidate target column names.

    Returns:
        Long-form table with correlations to each target column.
    """
    rows: List[pd.DataFrame] = []

    for target in target_candidates:
        if target not in numeric_columns:
            continue

        feature_cols = [col for col in numeric_columns if col != target]
        current = df.loc[:, feature_cols + [target]].corr(method="pearson")[target]
        current = current.drop(index=target).reset_index()
        current.columns = ["feature", "correlation"]
        current["abs_correlation"] = current["correlation"].abs()
        current["target"] = target
        current = current.sort_values("abs_correlation", ascending=False)
        rows.append(current)

    if not rows:
        return pd.DataFrame(columns=["target", "feature", "correlation", "abs_correlation"])

    result = pd.concat(rows, axis=0, ignore_index=True)
    return result.loc[:, ["target", "feature", "correlation", "abs_correlation"]]



def compute_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing value counts and percentages for all columns.

    Args:
        df: Input DataFrame.

    Returns:
        Summary table with missing counts and fractions.
    """
    summary = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_fraction": df.isna().mean(),
        "dtype": df.dtypes.astype(str),
    })
    return summary.sort_values(["missing_fraction", "missing_count"], ascending=False)



def compute_low_variance_summary(df: pd.DataFrame, numeric_columns: Sequence[str]) -> pd.DataFrame:
    """Compute variance and standard deviation per numeric feature.

    Args:
        df: Input DataFrame.
        numeric_columns: Numeric feature columns.

    Returns:
        Sorted table with variance statistics.
    """
    stats_df = pd.DataFrame({
        "std": df.loc[:, numeric_columns].std(),
        "variance": df.loc[:, numeric_columns].var(),
        "n_unique": df.loc[:, numeric_columns].nunique(),
    })
    return stats_df.sort_values(["variance", "std", "n_unique"], ascending=True)



def plot_histograms(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    output_path: Path,
    max_features: int,
) -> None:
    """Plot histograms for a subset of informative features.

    Args:
        df: Input DataFrame.
        numeric_columns: Numeric feature columns.
        output_path: Destination image path.
        max_features: Maximum number of features to visualize.
    """
    # Choose features with the highest variance because they are often most informative.
    variance_order = df.loc[:, numeric_columns].var().sort_values(ascending=False)
    selected = variance_order.head(max_features).index.tolist()

    if not selected:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), dpi=180)
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, selected):
        ax.hist(df[col].dropna(), bins=40)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle("Feature distributions (highest-variance features)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def plot_pca_explained_variance(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    output_path: Path,
    max_components: int = 50,
) -> pd.DataFrame:
    """Fit PCA and plot cumulative explained variance.

    Args:
        df: Input DataFrame.
        numeric_columns: Numeric feature columns.
        output_path: Destination image path.
        max_components: Maximum number of PCA components to evaluate.

    Returns:
        DataFrame containing explained variance information.
    """
    valid_df = df.loc[:, numeric_columns].dropna()
    if valid_df.empty:
        return pd.DataFrame(columns=["component", "explained_variance_ratio", "cumulative_explained_variance"])

    n_components = min(max_components, valid_df.shape[0], valid_df.shape[1])
    if n_components < 1:
        return pd.DataFrame(columns=["component", "explained_variance_ratio", "cumulative_explained_variance"])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(valid_df)

    pca = PCA(n_components=n_components)
    pca.fit(scaled)

    pca_df = pd.DataFrame({
        "component": np.arange(1, n_components + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    ax.plot(pca_df["component"], pca_df["cumulative_explained_variance"])
    ax.set_title("PCA cumulative explained variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return pca_df



def compute_average_participant_correlation(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    min_rows_per_participant: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the average within-participant correlation matrix.

    This is useful when global correlations may be inflated by concatenating many
    participants. The function computes one correlation matrix per participant and
    then averages them.

    Args:
        df: Input DataFrame with a ``__participant_id__`` helper column.
        numeric_columns: Numeric feature columns.
        min_rows_per_participant: Minimum row count required for a participant to
            be included.

    Returns:
        Tuple of:
        - averaged participant-wise correlation matrix
        - per-participant metadata table with row counts
    """
    participant_corrs: List[np.ndarray] = []
    metadata_rows: List[dict] = []

    for participant_id, group in df.groupby("__participant_id__"):
        current = group.loc[:, numeric_columns]
        current = current.dropna(axis=0, how="any")
        n_rows = len(current)
        metadata_rows.append({
            "participant_id": participant_id,
            "n_rows_used": n_rows,
            "included": n_rows >= min_rows_per_participant,
        })

        if n_rows < min_rows_per_participant:
            continue

        corr = current.corr(method="pearson")
        participant_corrs.append(corr.values)

    metadata_df = pd.DataFrame(metadata_rows).sort_values("participant_id")

    if not participant_corrs:
        empty = pd.DataFrame(np.nan, index=numeric_columns, columns=numeric_columns)
        return empty, metadata_df

    mean_corr = np.nanmean(np.stack(participant_corrs, axis=0), axis=0)
    mean_corr_df = pd.DataFrame(mean_corr, index=numeric_columns, columns=numeric_columns)
    return mean_corr_df, metadata_df



def compute_correlation_difference(train_corr: pd.DataFrame, val_corr: pd.DataFrame) -> pd.DataFrame:
    """Compute train-minus-validation correlation differences.

    Args:
        train_corr: Training correlation matrix.
        val_corr: Validation correlation matrix.

    Returns:
        Difference matrix aligned on common feature columns.
    """
    common = train_corr.index.intersection(val_corr.index)
    aligned_train = train_corr.loc[common, common]
    aligned_val = val_corr.loc[common, common]
    return aligned_train - aligned_val



def write_summary_text(
    output_path: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_corr: pd.DataFrame,
    val_corr: pd.DataFrame,
    train_participant_corr: pd.DataFrame,
    val_participant_corr: pd.DataFrame,
) -> None:
    """Write a short textual summary for report drafting.

    Args:
        output_path: Destination path for the summary text file.
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        train_corr: Global train correlation matrix.
        val_corr: Global validation correlation matrix.
        train_participant_corr: Averaged participant-wise train correlation matrix.
        val_participant_corr: Averaged participant-wise val correlation matrix.
    """
    def mean_abs_offdiag(corr_df: pd.DataFrame) -> float:
        values = corr_df.values.astype(float)
        mask = ~np.eye(values.shape[0], dtype=bool)
        return float(np.nanmean(np.abs(values[mask])))

    lines = [
        "EMAP feature EDA summary",
        "=" * 80,
        "",
        f"Train rows: {len(train_df)}",
        f"Validation rows: {len(val_df)}",
        f"Train participants: {train_df['__participant_id__'].nunique()}",
        f"Validation participants: {val_df['__participant_id__'].nunique()}",
        "",
        f"Mean absolute off-diagonal correlation (train, global): {mean_abs_offdiag(train_corr):.4f}",
        f"Mean absolute off-diagonal correlation (val, global): {mean_abs_offdiag(val_corr):.4f}",
        f"Mean absolute off-diagonal correlation (train, participant-wise average): {mean_abs_offdiag(train_participant_corr):.4f}",
        f"Mean absolute off-diagonal correlation (val, participant-wise average): {mean_abs_offdiag(val_participant_corr):.4f}",
        "",
        "Interpretation hint:",
        "If global correlations are much stronger than participant-wise average correlations,",
        "the overall heatmap is likely inflated by shared trends, baseline shifts, or the",
        "concatenation of multiple participant time series.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")



def run_analysis(config: AnalysisConfig) -> None:
    """Run the complete EDA pipeline.

    Args:
        config: Analysis configuration object.
    """
    sns.set_theme(style="white")
    root_dir, plots_dir, tables_dir = prepare_output_dirs(config.output_dir)
    text_dir = root_dir / "text"
    text_dir.mkdir(exist_ok=True)

    print("Loading training split...")
    train_df = load_split_dataframe(
        input_path=config.train_input,
        split_name="train",
        participant_regex=config.participant_regex,
        max_rows=config.max_rows_per_split,
        random_state=config.random_state,
    )

    print("Loading validation split...")
    val_df = load_split_dataframe(
        input_path=config.val_input,
        split_name="val",
        participant_regex=config.participant_regex,
        max_rows=config.max_rows_per_split,
        random_state=config.random_state,
    )

    # Restrict both splits to the same numeric features so that comparisons are fair.
    train_numeric = set(get_numeric_feature_columns(train_df))
    val_numeric = set(get_numeric_feature_columns(val_df))
    common_numeric = sorted(train_numeric.intersection(val_numeric))

    if not common_numeric:
        raise ValueError("No common numeric columns found between train and validation.")

    print(f"Common numeric features: {len(common_numeric)}")
    print(f"Train rows: {len(train_df)}, participants: {train_df['__participant_id__'].nunique()}")
    print(f"Val rows: {len(val_df)}, participants: {val_df['__participant_id__'].nunique()}")

    # Save raw descriptive statistics and structural summaries.
    train_df.loc[:, common_numeric].describe().T.to_csv(tables_dir / "train_descriptive_statistics.csv")
    val_df.loc[:, common_numeric].describe().T.to_csv(tables_dir / "val_descriptive_statistics.csv")
    compute_missing_value_summary(train_df).to_csv(tables_dir / "train_missing_values.csv")
    compute_missing_value_summary(val_df).to_csv(tables_dir / "val_missing_values.csv")
    compute_low_variance_summary(train_df, common_numeric).to_csv(tables_dir / "train_low_variance_features.csv")
    compute_low_variance_summary(val_df, common_numeric).to_csv(tables_dir / "val_low_variance_features.csv")

    # Global correlations across concatenated rows.
    print("Computing global correlations...")
    train_corr = compute_correlation_matrix(train_df, common_numeric, method="pearson")
    val_corr = compute_correlation_matrix(val_df, common_numeric, method="pearson")
    train_corr.to_csv(tables_dir / "train_correlation_pearson.csv")
    val_corr.to_csv(tables_dir / "val_correlation_pearson.csv")

    # Top pairwise correlations are useful for report text and redundancy analysis.
    compute_top_correlation_pairs(train_corr, config.top_corr_pairs).to_csv(
        tables_dir / "train_top_feature_feature_correlations.csv", index=False
    )
    compute_top_correlation_pairs(val_corr, config.top_corr_pairs).to_csv(
        tables_dir / "val_top_feature_feature_correlations.csv", index=False
    )

    # Feature-target correlations help explain which signals are most associated with the targets.
    compute_target_correlations(train_df, common_numeric, config.target_candidates).to_csv(
        tables_dir / "train_target_correlations.csv", index=False
    )
    compute_target_correlations(val_df, common_numeric, config.target_candidates).to_csv(
        tables_dir / "val_target_correlations.csv", index=False
    )

    print("Creating heatmaps...")
    plot_correlation_heatmap(
        train_corr,
        plots_dir / "train_correlation_heatmap_full.png",
        title="Train feature correlation heatmap (Pearson)",
        clustered=False,
    )
    plot_correlation_heatmap(
        val_corr,
        plots_dir / "val_correlation_heatmap_full.png",
        title="Validation feature correlation heatmap (Pearson)",
        clustered=False,
    )

    print(f"Done. Outputs saved to: {root_dir}")



def main() -> None:
    """Configure and run the analysis.

    Edit the paths below once and then simply run the script without passing any
    command-line arguments.
    """
    config = AnalysisConfig(
        # ------------------------------------------------------------------
        # IMPORTANT: Adjust these paths to your local setup.
        # You can provide either a directory containing CSV files or a single
        # CSV file path.
        # ------------------------------------------------------------------
        train_input=r"data/train",
        val_input=r"data/val",
        output_dir=r"correlations/output",
        # Optional row cap for faster experimentation. Set to None for full data.
        max_rows_per_split=50000,
        random_state=42,
    )

    run_analysis(config)


if __name__ == "__main__":
    main()
