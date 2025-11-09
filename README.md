# Weekly Store Sales Forecasting

This project provides a solution for forecasting weekly sales for a retail chain. It includes a complete machine learning pipeline from data processing and model training to deployment via a REST API and an interactive web dashboard.

The core of the solution is a **LightGBM model** trained on historical sales data. This model is served by a **FastAPI** backend, and users can interact with it through a **Streamlit** web application. The entire stack is containerized using **Docker** for easy setup and deployment.

##  Features

- **Data Processing & Feature Engineering**: Robust script to transform raw data, creating cyclical time-based features and holiday proximity metrics.
- **High-Performance Model**: Utilizes a tuned LightGBM model to capture complex, non-linear patterns in sales data.
- **REST API**: A FastAPI backend exposes a `/predict` endpoint to get sales predictions from input data.
- **Interactive Dashboard**: A Streamlit application provides an intuitive user interface to upload data, view predictions, analyze model performance, and visualize sales trends.
- **Containerized**: The entire application (training, backend, frontend) is managed by Docker and Docker Compose for one-command setup.

## Project Structure

The project is organized into distinct modules for clarity and maintainability:

```
YOOBIC_TEST/
├── data/
│   ├── stores-sales.csv          # Raw training data
│   └── Prediction_test.csv       # Sample data for making predictions
├── model/
│   └── lgbm_sales_model.joblib   # Saved, trained model artifact
├── notebooks/
│   └── weekly_sales_forecasting.ipynb # Jupyter notebook for EDA, model experimentation, and tuning
├── src/
│   ├── __init__.py
│   ├── config.py                 # Central configuration for paths, parameters, etc.
│   ├── predict.py                # Logic for making predictions with the loaded model
│   ├── process.py                # Data preprocessing and feature engineering functions
│   └── train.py                  # Script to train and save the final model
├── .gitignore
├── api.py                        # FastAPI application to serve the model
├── app.py                        # Streamlit frontend application (dashboard)
├── docker-compose.yml            # Defines and orchestrates the multi-container services
├── Dockerfile                    # Instructions to build the application's Docker image
├── requirements.txt              # Python package dependencies
└── README.md                     # This file
```

## Modeling Approach

The model was developed following a structured, iterative process detailed in `notebooks/weekly_sales_forecasting.ipynb`.

1.  **Exploratory Data Analysis (EDA)**: The initial analysis revealed strong seasonality, significant sales variations between stores, and non-linear relationships between features and the target (`weekly_sales`). This guided the decision to use a tree-based model over linear alternatives.
2.  **Feature Engineering**: To help the model learn effectively, several features were engineered:
    - **Cyclical Features**: Date components (month, day, week of year) were transformed using `sin` and `cos` functions to represent their cyclical nature (e.g., December is close to January).
    - **Holiday Proximity**: Features like `weeks_to_christmas` were created to capture the sales build-up or decline around major holidays, which is more informative than a simple boolean `holiday_flag`.
3.  **Model Selection**:
    - **Baselines**: Linear Regression and Ridge models were tested, confirming that linear approaches were insufficient for this problem.
    - **Advanced Models**: Gradient Boosting models (LightGBM and XGBoost) were evaluated. Both performed exceptionally well, significantly outperforming the baselines.
    - **Hyperparameter Tuning**: A Grid Search was performed on the LightGBM model to find the optimal hyperparameters.
4.  **Final Model**: The **tuned LightGBM model** was selected as the final model due to its superior performance, achieving a **final Test RMSE of $79,824.43**. The target variable (`weekly_sales`) was log-transformed (`np.log1p`) during training to stabilize variance, and predictions are inverse-transformed back to their original scale.

## Setup and Execution

You can run this project using Docker (recommended) or by setting up a local Python environment.

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local setup)

### Method 1: Docker & Docker Compose (Recommended)

This is the simplest way to get the entire application running.

1.  **Clone the repository:**
    ```bash
    git clone https://https://github.com/OussamaLafdil/Yoobic_ML_Test
    cd YOOBIC_TEST
    ```

2.  **Build and run the services:**
    ```bash
    docker-compose up --build
    ```
    This command will:
    - Build the Docker image based on the `Dockerfile`.
    - Run the `training` service to train the model and save it to the `model/` directory.
    - Start the `backend` FastAPI service.
    - Start the `frontend` Streamlit service.

3.  **Access the applications:**
    - **Streamlit Dashboard**: Open your browser and go to `http://localhost:8501`
    - **FastAPI Backend (API Docs)**: Open your browser and go to `http://localhost:8000/docs`

### Method 2: Local/Manual Setup

If you prefer not to use Docker, follow these steps.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd YOOBIC_TEST
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the training script:**
    This will create the `lgbm_sales_model.joblib` file in the `model/` directory.
    ```bash
    python src/train.py
    ```

5.  **Important: Configure the backend URL:**
    Before starting the services, open `src/config.py` and change the `BACKEND_URL` for local use:
    ```python
    # In src/config.py
    BACKEND_URL = "http://localhost:8000" # FROM "http://backend:8000"
    ```

6.  **Run the services (in separate terminals):**
    - **Terminal 1: Start the FastAPI Backend**
      ```bash
      uvicorn api:app --host 0.0.0.0 --port 8000
      ```
    - **Terminal 2: Start the Streamlit Frontend**
      ```bash
      streamlit run app.py
      ```

7.  **Access the applications:**
    - **Streamlit Dashboard**: `http://localhost:8501`
    - **FastAPI Backend (API Docs)**: `http://localhost:8000/docs`

##  How to Use the Application

The primary way to interact with the model is through the Streamlit web dashboard.

1.  **Navigate to the Dashboard**: Open `http://localhost:8501` in your web browser.
2.  **Upload Data**: Use the file uploader in the sidebar to upload a `.csv` or `.xlsx` file containing store data for prediction. You can use the provided `data/Prediction_test.csv` as an example.
3.  **Get Predictions**: The application sends your data to the FastAPI backend. A spinner will indicate that the prediction is in progress.
4.  **View Results**: Once complete, the results are displayed in a table, showing your original data with a new `predicted_sales` column.
5.  **Analyze and Visualize**:
    - If your uploaded data includes the actual `weekly_sales` column, the dashboard will calculate and display the RMSE to show model performance on your data.
    - Use the dropdown menu to select a specific store and visualize the trend of predicted sales (and actual sales, if available) over time.
6.  **Download Predictions**: Click the "Download Predictions as CSV" button to save the results to your local machine.

### API Endpoint

For programmatic access, you can send a `POST` request directly to the FastAPI endpoint.

-   **Endpoint**: `/predict`
-   **Method**: `POST`
-   **Body**: `multipart/form-data` with a `file` key containing your `.csv` or `.xlsx` file.


## Web Application Demo

Below is a brief demonstration of the Streamlit dashboard. The demo shows how a user can upload a test data file, viewing the sales predictions in a table, visualizing the results for a specific store, and downloading the output.

![Web App Demo](https://raw.githubusercontent.com/OussamaLafdil/Yoobic_ML_Test/assets//Démo_UI.mp4)