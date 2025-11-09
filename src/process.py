import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import  DATE_COLUMN, HOLIDAYS, COLS_TO_DROP_POST_FE

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all preprocessing and feature engineering steps to the raw sales data.

    Args:
        df (pd.DataFrame): The raw original input DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame ready for model training or prediction.
    """
    processed_df = df.copy()

    # --- 1. Date Conversion ---
    processed_df[DATE_COLUMN] = pd.to_datetime(processed_df[DATE_COLUMN], format='%d-%m-%Y')

    # --- 2. Basic Date Features ---
    processed_df['year'] = processed_df[DATE_COLUMN].dt.year
    processed_df['month'] = processed_df[DATE_COLUMN].dt.month
    processed_df['day'] = processed_df[DATE_COLUMN].dt.day
    processed_df['week_of_year'] = processed_df[DATE_COLUMN].dt.isocalendar().week.astype(int)
    processed_df['day_of_year'] = processed_df[DATE_COLUMN].dt.dayofyear
    processed_df['quarter'] = processed_df[DATE_COLUMN].dt.quarter

    # --- 3. Cyclical Date Features ---
    processed_df['days_in_month'] = processed_df[DATE_COLUMN].dt.days_in_month
    processed_df['is_leap'] = processed_df[DATE_COLUMN].dt.is_leap_year
    processed_df['days_in_year'] = processed_df['is_leap'].apply(lambda x: 366 if x else 365)

    # Month
    processed_df['month_sin'] = np.sin(2 * np.pi * processed_df['month'] / 12)
    processed_df['month_cos'] = np.cos(2 * np.pi * processed_df['month'] / 12)
    # Week of year
    processed_df['week_of_year_sin'] = np.sin(2 * np.pi * processed_df['week_of_year'] / 53)
    processed_df['week_of_year_cos'] = np.cos(2 * np.pi * processed_df['week_of_year'] / 53)
    # Day within the month
    processed_df['day_sin'] = np.sin(2 * np.pi * processed_df['day'] / processed_df['days_in_month'])
    processed_df['day_cos'] = np.cos(2 * np.pi * processed_df['day'] / processed_df['days_in_month'])
    # Day in year
    processed_df['day_of_year_sin'] = np.sin(2 * np.pi * processed_df['day_of_year'] / processed_df['days_in_year'])
    processed_df['day_of_year_cos'] = np.cos(2 * np.pi * processed_df['day_of_year'] / processed_df['days_in_year'])

    # --- 4. Holiday Proximity Features ---
    for holiday_name, holiday_dates_str in HOLIDAYS.items():
        holiday_dates = pd.to_datetime(holiday_dates_str)
        year_to_holiday_date = {d.year: d for d in holiday_dates}
        holiday_date_for_row = processed_df['year'].map(year_to_holiday_date)
        
        # Calculate weeks to/from the holiday
        processed_df[f'weeks_to_{holiday_name}'] = (processed_df[DATE_COLUMN] - holiday_date_for_row).dt.days / 7
        
        # Fill missing values for years where the holiday date isn't defined
        # A large number signifies the event is far away.
        processed_df[f'weeks_to_{holiday_name}'].fillna(99, inplace=True)

    # --- 5. Cleanup ---
    processed_df = processed_df.drop(columns=COLS_TO_DROP_POST_FE, errors='ignore')

    return processed_df


# Simple test : 

# original_df = pd.read_csv(DATA_PATH)

# featured_df = process_data(original_df)

# print("DataFrame with new date features:")
# print(featured_df)