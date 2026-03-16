"""
Compute per-patient feature correlation heatmaps for train and validation data.

This script:
1. Reads all CSV files from TRAIN_PATH and VAL_PATH.
2. Extracts the participant ID from the filename (pattern: PXXX).
3. Computes a Pearson correlation matrix for each participant.
4. Randomly selects N participants from train and val.
5. Saves correlation heatmaps for these participants.

Expected filename pattern:
    Features_P001-T01.csv

The script assumes that each CSV contains the same feature columns.
"""

import os
import glob
import random
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# =============================================================================
# Paths (edit here if needed)
# =============================================================================

TRAIN_PATH = "data/train"
VAL_PATH = "data/val"

OUTPUT_DIR = "correlations/output"

N_RANDOM_PATIENTS = 5

# =============================================================================
# Utility functions
# =============================================================================

def parse_participant_id(filename: str) -> str:
    """
    Extract participant ID from filename.

    Args:
        filename: Path to CSV file.

    Returns:
        Participant ID string.
    """
    basename = os.path.basename(filename)
    match = re.search(r"P(\d{3})", basename)

    if match is None:
        return "unknown"

    return match.group(1)


def load_participant_data(path: str) -> Dict[str, List[pd.DataFrame]]:
    """
    Load CSV files and group them by participant.

    Args:
        path: Folder containing CSV files.

    Returns:
        Dictionary mapping participant_id -> list of DataFrames.
    """
    files = sorted(glob.glob(os.path.join(path, "*.csv")))

    participant_data = {}

    for file in files:
        pid = parse_participant_id(file)

        df = pd.read_csv(file, encoding="ISO-8859-1")

        if pid not in participant_data:
            participant_data[pid] = []

        participant_data[pid].append(df)

    return participant_data


def compute_patient_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for features.

    Args:
        df: Input DataFrame.

    Returns:
        Correlation matrix DataFrame.
    """
    df = df.copy()

    # Fill missing values
    df = df.fillna(0)

    return df.corr(method="pearson")


def save_heatmap(corr: pd.DataFrame, title: str, output_path: str) -> None:
    """
    Save correlation heatmap.

    Args:
        corr: Correlation matrix.
        title: Plot title.
        output_path: File path to save the image.
    """
    plt.figure(figsize=(14, 12))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading train data...")
    train_data = load_participant_data(TRAIN_PATH)

    print("Loading validation data...")
    val_data = load_participant_data(VAL_PATH)

    # Randomly sample participants
    random.seed(42)

    train_patients = random.sample(
        list(train_data.keys()),
        min(N_RANDOM_PATIENTS, len(train_data))
    )

    val_patients = random.sample(
        list(val_data.keys()),
        min(N_RANDOM_PATIENTS, len(val_data))
    )

    print("Selected train patients:", train_patients)
    print("Selected val patients:", val_patients)

    # -------------------------------------------------------------------------
    # Train heatmaps
    # -------------------------------------------------------------------------

    for pid in train_patients:

        df = pd.concat(train_data[pid], ignore_index=True)

        corr = compute_patient_correlation(df)

        output_file = os.path.join(
            OUTPUT_DIR,
            f"train_patient_{pid}_correlation.png"
        )

        save_heatmap(
            corr,
            f"Train correlation heatmap - Patient {pid}",
            output_file
        )

    # -------------------------------------------------------------------------
    # Validation heatmaps
    # -------------------------------------------------------------------------

    for pid in val_patients:

        df = pd.concat(val_data[pid], ignore_index=True)

        corr = compute_patient_correlation(df)

        output_file = os.path.join(
            OUTPUT_DIR,
            f"val_patient_{pid}_correlation.png"
        )

        save_heatmap(
            corr,
            f"Validation correlation heatmap - Patient {pid}",
            output_file
        )

    print("Done. Heatmaps saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()