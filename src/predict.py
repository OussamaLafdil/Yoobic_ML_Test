import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.process import process_data
from src.config import MODEL_PATH, DATA_PREDICTION_PATH, TARGET_COLUMN

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
    model = None

def make_predictions(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe with store sales data and returns it with an added 'predicted_sales' column.
    
    Args:
        input_df (pd.DataFrame): DataFrame with the same structure as the training data,
                                 (minus the 'weekly_sales' target).
    
    Returns:
        pd.DataFrame: The original DataFrame with a new 'predicted_sales' column.
                      Returns None if the model is not loaded.
    """
    if model is None:
        print("Prediction cannot be made because the model is not loaded.")
        return None
        
    print("Starting prediction process...")
    
    
    # --- 1. Process Input Data ---
    # The process_data function is reused here to ensure consistency
    processed_df = process_data(input_df)
    
    if TARGET_COLUMN in processed_df.columns:
        print(f"Warning: Target column '{TARGET_COLUMN}' found in input data. It will be ignored for prediction.")
        processed_df = processed_df.drop(columns=[TARGET_COLUMN])

    # --- 2. Select Features ---
    # Ensure the columns are in the same order as during training
    model_features = model.feature_name_

    missing = [f for f in model_features if f not in processed_df.columns]
    if missing:
        raise ValueError(f"Missing expected features: {missing}")
    

    X_pred = processed_df[model_features]
    
    # --- 3. Make Predictions ---
    print(f"Predicting for {len(X_pred)} rows...")
    pred_log = model.predict(X_pred)
    
    # Inverse transform to get actual sales values
    predictions = np.expm1(pred_log)
    
    # --- 4. Format Output ---
    output_df = input_df.copy()
    output_df['predicted_sales'] = predictions
    
    print("Prediction complete.")
    
    return output_df

if __name__ == '__main__':
    print("\n--- Running example prediction ---")
    
    try:
        pred_df = pd.read_csv(DATA_PREDICTION_PATH)
        
        print("Prediction input data:")
        print(pred_df)
        
        predictions_df = make_predictions(pred_df)
        
        if predictions_df is not None:
            print("\nPredictions:")
            print(predictions_df[['store', 'date', 'predicted_sales']])
            
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PREDICTION_PATH} for example prediction.")
    except Exception as e:
        print(f"An error occurred during the example run: {e}")