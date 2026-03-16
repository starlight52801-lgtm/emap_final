# EMAP Challenge - Physiological Signal Prediction

This repository contains experiments conducted for the EMAP Dataset Challenge, focusing on predicting physiological responses and emotional arousal from neurophysiological data.

The implemented approaches include classical machine learning models, deep learning models, and symbolic regression using Genetic Programming. 

The main prediction targets are:

- Heart Rate
- Galvanic Skin Response (GSR)
- Emotional Arousal

The models are evaluated using RMSE and NRMSE.

## Repository Structure

### correlations/
Contains scripts used for exploratory data analysis, specifically correlation analysis of the feature space. These scripts generate correlation matrices for:
- the complete training dataset
- individual participants

### preditions/
Contains all scripts used for model training and prediction for the three regression targets:
- Heart Rate
- Galvanic Skin Response (GSR)
- Emotional Arousal

Different modeling approaches are implemented, including:
- Genetic Programming
- LSTM networks
- classical machine learning models

Each script generates prediction results and evaluation metrics. 

## Installation
Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Running the scripts
All scripts must be executed from the repository root directory.

Correct:
```bash
python3 correlations/correlation_analysis_all.py
```

```bash
python3 predictions/further_models.py
```

## Dataset
The scripts expect the dataset to be organized as:
```bash
data/
├── train/
└── val/
```
The repository contains **placeholder files only**.
The original EMAP dataset is not included due to storage limitations.


## Evaluation
Model performance is evaluated using:
- Root Mean Squared Error (RMSE)
- Normalized Root Mean Squared Error (NRMSE)

Prediction plots are generated for validation participants to visually compare predicted and ground truth signals.

## Notes

- Some models require GPU acceleration when available.