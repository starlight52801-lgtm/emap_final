"""
PV temporal GP regressor
This script implements a 24-fold cross-validation as described in the original paper over a single patient using a Genetic Programming Regressor.
Targets:
    - heartrate_mean
    - GSR_mean
    - LABEL_SR_Arousal
Note: Ensure that the required libraries are installed and the data paths are correctly set before running the script.
"""


from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler


TRAIN_PATH = Path("./data/train")  # Folder containing training CSV files
patient = "P001" # filter substring for patient CSVs
target = "LABEL_SR_Arousal" # one of ["GSR_mean" "GSR_mean" "heartrate_mean"]
output_plot = Path(f"./plots/pv_for_patient_{patient}_and_{target}.png") # output plot path
results_csv_path = Path("./predictions/output/validation_results_{target}.csv") # output csv path for validation results
program_csv_path = Path(f"./predictions/output/best_program_{target}.csv") # output csv path for metrics

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



def clean_csv(input_csv, target):
    input_csv['heartrate_mean'] = input_csv['heartrate_mean'] - input_csv['heartrate_mean'].mean()
    return input_csv

def add_lag(df, targets):
    """Add lagged features"""
    df = df.copy()
    lagged_cols = []

    for feature in df.columns.values.tolist():
        if feature not in targets:
            #print(f'Currently processing feature: {feature}')
            for lag in [1, 2, 5, 10]:
                new_col = df[feature].shift(lag).rename(f"{feature}{lag}")
                df[feature] = new_col
    df = df.bfill().fillna(df.mean(numeric_only=True))
    return df

def get_all_cvs_for_substring(folder_path, substring, target):
    """
    Finds all CSV files matching a substring, combines data, and creates a
    single line plot
    Args:
        folder_path (str): The path to the folder containing the CSV files
        substring (str): The required substring to filter
    """

    search_pattern = os.path.join(folder_path, f"*{substring}*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No CSV files found in '{folder_path}' containing the substring '{substring}'.")
        return

    print(f"Found {len(csv_files)} files to plot against the common index.")

    all_data_frames = []
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            #df = df.fillna(0)
            #df = add_lag(df, targets=[target])
            all_data_frames.append(df)
        except Exception as e:
            print(f"Error reading or processing file {file_name}: {e}")
    return all_data_frames

all_csvs = get_all_cvs_for_substring(TRAIN_PATH, patient, target)

all_predictions = []
all_actuals = []

for i in range(24):
    print(f"Processing Fold {i + 1}/24...")

    test_df = all_csvs[i]
    train_df = pd.concat([all_csvs[j] for j in range(24) if j != i])
    if filter:
        train_df = train_df[kept_features]
        test_df = test_df[kept_features]

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[[target]]

    model = SymbolicRegressor(
    generations=50,
    population_size=3000,
    tournament_size=5,
    init_depth=(2, 6),
    #function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'sin', 'cos'],
    #function_set=['add', 'sub', 'mul', 'log', 'abs', 'neg', 'inv', 'sin', 'cos'],
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log'],
    p_crossover=0.8,
    p_subtree_mutation= 0.1,
    p_hoist_mutation= 0.05,
    p_point_mutation= 0.05,
    random_state=42,
    parsimony_coefficient= 0.0001,
    verbose=1,
    low_memory = True
)
    #y_train_mean = y_train.mean().item()
    #y_train_centered = y_train - y_train_mean
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train).ravel()
    model.fit(X_train, y_train_scaled)
    #model.fit(X_train, y_train_centered)
    #model.fit(X_train, y_train)


    preds_scaled = model.predict(X_test)
    #preds = model.predict(X_test)

    preds_unscaled = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    #preds = preds_unscaled
    #preds = preds_unscaled + y_train_mean
    #preds = preds + y_train_mean
    all_predictions.extend(preds_unscaled)

    #all_predictions.extend(preds)
    all_actuals.extend(test_df[target].tolist())


actuals_arr = np.array(all_actuals)
preds_arr = np.array(all_predictions)


mae = mean_absolute_error(actuals_arr, preds_arr)
rmse = np.sqrt(mean_squared_error(actuals_arr, preds_arr))
nrmse = rmse / (actuals_arr.max() - actuals_arr.min())
r2 = r2_score(actuals_arr, preds_arr)


print("Metrics")
print(f"MAE:   {mae}")
print(f"RMSE:  {rmse}")
print(f"NRMSE: {nrmse} (or {nrmse}%)")


#Save Results
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results_df.to_csv(results_csv_path, index=False)
print(f"Validation results saved to: {results_csv_path}")

#Save Best Program
best_program_str = str(model._program)

program_data = {
    'Target': [target],
    'RMSE': [rmse],
    'NRMSE': [nrmse],
    'Equation': [best_program_str]
}

program_df = pd.DataFrame(program_data)
program_df.to_csv(program_csv_path, index=False)

plt.figure(figsize=(15, 6))
plt.plot(all_actuals, label='Actual Values', alpha=0.7, color='blue')
plt.plot(all_predictions, label='GP Predictions', alpha=0.7, color='orange', linestyle='--')
plt.title('Patient Time Frames: Actual vs. Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.savefig(output_plot, dpi=300)
plt.close()