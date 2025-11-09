import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_PATH, DATE_COLUMN, VAL_SPLIT_DATE, TEST_SPLIT_DATE, TARGET_COLUMN, LGBM_PARAMS, EARLY_STOPPING_ROUNDS, CATEGORICAL_FEATURES, MODEL_PATH
from src.process import process_data

def run_training():
    """
    Loads data, processes it, trains a LightGBM model with the best hyperparmeters, and saves it.
    """
    print("--- Starting Training Pipeline ---")

    # --- 1. Load Data ---
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # --- 2. Process Data ---
    print("Processing data and engineering features...")
    processed_df = process_data(df)

    # --- 3. Prepare for LightGBM ---
    FEATURES = [col for col in processed_df.columns if col not in [DATE_COLUMN, TARGET_COLUMN]]

    X_train = processed_df[FEATURES]
    y_train = processed_df[TARGET_COLUMN]

    # Log-transform target variable
    y_train_log = np.log1p(y_train)
    
    # --- 4. Train Model ---
    print("\nTraining LightGBM model...")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    model.fit(
        X_train, y_train_log,
        categorical_feature=CATEGORICAL_FEATURES
    )
    
    print("Model training complete on the full dataset.")

    # --- 5. Save Model ---
    print(f"Saving trained model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully.")
    print("\n--- Training Pipeline Finished ---")

if __name__ == '__main__':
    run_training()