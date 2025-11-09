
# --- File Paths ---
DATA_PATH = "data/stores-sales.csv" 
DATA_PREDICTION_PATH = "data/Prediction_test.csv"  

MODEL_PATH = "model/lgbm_sales_model.joblib" 

# --- Data Columns ---
TARGET_COLUMN = 'weekly_sales'
DATE_COLUMN = 'date'

# Holiday dates within the dataset's timeframe
HOLIDAYS = {
    'super_bowl': ['2010-02-12', '2011-02-11', '2012-02-10'],
    'labour_day': ['2010-09-10', '2011-09-09', '2012-09-07'],
    'thanksgiving': ['2010-11-26', '2011-11-25', '2012-11-23'],
    'christmas': ['2010-12-31', '2011-12-30', '2012-12-28']
}

# columns to drop
COLS_TO_DROP_POST_FE = ['days_in_year', 'is_leap', 'days_in_month', 'month', 'day', 'week_of_year', 'day_of_year']

# Train/Validation/Test split dates
VAL_SPLIT_DATE = '2012-01-01'
TEST_SPLIT_DATE = '2012-06-01'

# Categorical features
CATEGORICAL_FEATURES = ['store']

# LightGBM Hyperparameters 
LGBM_PARAMS = {
    'objective': 'regression_l1', # MAE
    'metric': 'rmse',
    'n_estimators': 2599,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': -1,
    'reg_alpha': 0.5,
    'reg_lambda': 0.1,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# Early stopping rounds
EARLY_STOPPING_ROUNDS = 100

#Backend url : 
BACKEND_URL = "http://backend:8000" #change to "http://localhost:8000" if you are not using the docker compose

