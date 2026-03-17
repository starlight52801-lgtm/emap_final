"""
Temporal GP regressor for EMAP CSV files using gplearn.
Smaller version of the main Temporal GP regressor script (gp.py) with reduced features and without a cross validation.
Mail purpose is to test if the GP regressor can learn any meaningful patterns given the available base features, and to get a first impression of the best program found by the GP.

Targets:
- "heartrate"
- "gsr"
- "arousal"

Notes:
- Only CPU is supported by gplearn
- No parallelization 
- The final RMSE / NRMSE are computed on the original target scale.
- Ensure that the required libraries are installed and the data paths are correctly set before running the script.
"""

import pandas as pd
import glob
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path

target = "heartrate_mean" # one of [LABEL_SR_Arousal, GSR_mean, heartrate_mean]
TRAIN_PATH = Path("./data/train")  # Folder containing training CSV files
VAL_PATH = Path("./data/val")  # Folder containing validation CSV files
MODEL_FILENAME = Path("./predictions/output/gpr_best_model_{target}.joblib")
results_csv_path = Path("./predictions/output/validation_results_{target}.csv")
program_csv_path = Path(f"./predictions/output/best_program_{target}.csv")

# used to test if feature slecetion would yield an improvement, keep false to include all features
filter = False
kept_features = ["EEG_AF3_Theta",
"EEG_AF8_Theta",
"EEG_C5_Alpha",
"EEG_C5_Theta",
"EEG_CP1_Theta",
"EEG_CP2_Theta",
"EEG_CPz_Theta",
"EEG_F4_Alpha",
"EEG_F4_Theta",
"EEG_F6_Theta",
"EEG_F7_Alpha",
"EEG_F7_Gamma",
"EEG_FC1_Theta",
"EEG_FC5_Theta",
"EEG_FT10_Alpha",
"EEG_FT10_Theta",
"EEG_Fp1_Theta",
"EEG_Fp2_Gamma",
"EEG_Fp2_Theta",
"EEG_O1_Theta",
"EEG_O2_Alpha",
"EEG_O2_Theta",
"EEG_Oz_Theta",
"EEG_PO3_Theta",
"EEG_PO4_Theta",
"EEG_POz_Theta",
"EEG_T7_Theta",
"EEG_TP10_Theta",
"EEG_TP7_Theta",
"EEG_TP8_Theta",
"EEG_TP9_Theta",
"GSR_mean",
"IRPleth_mean",
"LABEL_SR_Arousal",
"Respir_mean",
"heartrate_mean"]


def load_data_from_path(path, phase_name):
    """Helper function to load all CSVs from a given path"""
    print(f"Reading data from: {path} for {phase_name}...")
    all_files = glob.glob(os.path.join(path, "*.csv"))

    if not all_files:
        print(f"WARNING: No CSV files found in {path}. Check path and file names.")
        return pd.DataFrame()

    li = []
    for filename in all_files:
        df = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
        li.append(df)

    dataset = pd.concat(li, axis=0, ignore_index=True)
    dataset = dataset.fillna(0)
    print(f"{phase_name} data loaded. Shape: {dataset.shape}")
    return dataset

def plot_predictions(y_true, y_pred, title, out_path):
    """
    Plot True vs Predicted
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="True", alpha=0.7)
    plt.plot(y_pred, label="Predicted", alpha=0.7, linestyle='--')
    plt.title(title)
    plt.xlabel("Trial Index (Validation Set)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

train_dataset = load_data_from_path(TRAIN_PATH, "Training Set")
val_dataset = load_data_from_path(VAL_PATH, "Validation Set")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

if filter:
    train_dataset = train_dataset[kept_features]
    val_dataset = val_dataset[kept_features]

if not train_dataset.empty:

    # Define features
    X_train_full = train_dataset.iloc[:, train_dataset.columns != target]
    y_train_full = train_dataset.iloc[:, train_dataset.columns == target].values
    feature_cols = X_train_full.columns.tolist()
    y_train_full = scaler_y.fit_transform(y_train_full)

    X_val_full = val_dataset.iloc[:, val_dataset.columns != target]
    X_val_full = X_val_full[feature_cols]
    y_val_full = val_dataset.iloc[:, val_dataset.columns == target].values

    print(f"Final features count: {len(feature_cols)}")

    X_train_gp = X_train_full
    y_train = y_train_full.flatten()
    X_test_gp = X_val_full
    y_test = y_val_full.flatten()

else:
    print("Training data loading failed. Exiting script.")
    exit()

gpr = SymbolicRegressor(
    generations=50,
    population_size=3000,
    tournament_size=15,
    init_depth=(2, 6),
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log'],
    feature_names=feature_cols,
    parsimony_coefficient=0.00005,
    p_crossover=0.8,
    p_subtree_mutation= 0.1,
    p_hoist_mutation= 0.05,
    p_point_mutation= 0.05,
    random_state=42,
    verbose=1,
    warm_start=False,
    low_memory = True,
    metric='rmse'
)

print(f"\nStarting GP Regressor Training on Training Set for {target}...")
gpr.fit(X_train_gp, y_train)
print("GP Training Complete.")

y_pred = gpr.predict(X_test_gp)
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Performance on Validation Set for {target}---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}, RMSE_func:{root_mean_squared_error(y_test, y_pred)}")
print(f"R-squared (R2): {r2:.4f}")

y_range = np.max(y_test) - np.min(y_test)
nrmse = rmse / y_range if y_range != 0 else 0

#Save Results
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results_df.to_csv(results_csv_path, index=False)
print(f"Validation results saved to: {results_csv_path}")

#Save Best Program
best_program_str = str(gpr._program)

program_data = {
    'Target': [target],
    'RMSE': [rmse],
    'NRMSE': [nrmse],
    'R2': [r2],
    'Equation': [best_program_str]
}

program_df = pd.DataFrame(program_data)
program_df.to_csv(program_csv_path, index=False)

print(f"\n--- Performance Metrics ---")
print(f"RMSE: {rmse:.4f}")
print(f"NRMSE: {nrmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"Best program details saved to: {program_csv_path}")

best_equation = gpr._program

print("\n--- Best Equation Found ---")
print(best_equation)
print("---------------------------\n")

#Save Model
try:
    joblib.dump(gpr, MODEL_FILENAME)
    print(f"Successfully saved the trained GP model to: {MODEL_FILENAME}")
except Exception as e:
    print(f"Error saving model: {e}")