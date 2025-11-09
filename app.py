import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
import os
from sklearn.metrics import mean_squared_error

from src.config import TARGET_COLUMN, BACKEND_URL

# --- 1. APP CONFIG ---
st.set_page_config(
    page_title="Store Sales Forecaster",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"

# --- HELPER FUNCTIONS ---
@st.cache_data
def convert_df_to_csv(df):
    """Caches the conversion of a DataFrame to CSV to improve performance."""
    return df.to_csv(index=False).encode('utf-8')

def get_predictions(upload_file):
    """Sends the uploaded file to the backend API and returns the predictions."""
    files = {'file': (upload_file.name, upload_file.getvalue(), upload_file.type)}
    
    try:
        response = requests.post(PREDICT_ENDPOINT, files=files, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        response_json = response.json()
        
        if "predictions" in response_json:
            return pd.DataFrame(response_json["predictions"])
        else:
            st.error(f"Prediction failed: {response_json.get('detail', 'Unknown error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the prediction API: {e}")
        st.info(f"Please ensure the backend service is running at {BACKEND_URL}")
        return None

# --- 2. MAIN APPLICATION ---
def main():
    """The main function that runs the Streamlit application."""
    
    # --- SIDEBAR ---
    st.sidebar.title("🛒 Sales Forecaster")
    st.sidebar.markdown("Upload your store data to generate weekly sales predictions.")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or XLSX file",
        type=['csv', 'xlsx'],
        help="The file should contain columns like 'store', 'date', 'temperature', etc."
    )
    
    # Initialize session state
    if 'predictions_df' not in st.session_state:
        st.session_state['predictions_df'] = None

    # --- MAIN CONTENT ---
    st.title("Weekly Store Sales Prediction")
    st.markdown("This web app uses a LightGBM model served via a FastAPI backend to predict weekly sales. Upload a file to get started.")

    if uploaded_file is not None:
        st.sidebar.success(f"File '{uploaded_file.name}' uploaded. Processing...")

        # --- Generate Predictions via API Call ---
        with st.spinner('Sending data to the model API... Please wait.'):
            predictions_df = get_predictions(uploaded_file)
            if predictions_df is not None:
                st.session_state['predictions_df'] = predictions_df # Store in session state

    # --- Display results if they exist in the session state ---
    if st.session_state['predictions_df'] is not None:
        results_df = st.session_state['predictions_df']
        
        # --- 3. RESULTS DISPLAY ---
        st.header("Prediction Results")
        st.markdown("Below are the predicted sales for each store and week in your dataset.")
        
        st.dataframe(results_df, use_container_width=True)

        # --- 4. DOWNLOAD OPTION ---
        csv_data = convert_df_to_csv(results_df)
        st.download_button(
           label="📥 Download Predictions as CSV",
           data=csv_data,
           file_name="predicted_sales.csv",
           mime="text/csv",
        )

        st.markdown("---") 

        # --- 5. BONUS FEATURES ---
        st.header("Analysis & Visualization")

        if TARGET_COLUMN in results_df.columns:
            st.subheader("Model Performance")
            
            valid_results = results_df.dropna(subset=[TARGET_COLUMN, 'predicted_sales'])
            rmse = np.sqrt(mean_squared_error(valid_results[TARGET_COLUMN], valid_results['predicted_sales']))
            
            col1, col2 = st.columns(2)
            col1.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
            col2.info("RMSE measures the average magnitude of the prediction errors.")
            st.markdown("") 
        
        st.subheader("Explore Predictions by Store")
        
        stores = sorted(results_df['store'].unique())
        selected_store = st.selectbox(
            "Select a store to visualize its sales trend:",
            options=stores
        )
        
        if selected_store:
            store_df = results_df[results_df['store'] == selected_store].copy()
            store_df['date'] = pd.to_datetime(store_df['date'], format='%d-%m-%Y')
            store_df = store_df.sort_values('date')

            fig_data_cols = ['predicted_sales']
            if TARGET_COLUMN in store_df.columns:
                fig_data_cols.append(TARGET_COLUMN)
                
            fig_data = store_df.melt(id_vars=['date'], value_vars=fig_data_cols, var_name='Sales Type', value_name='Sales')
            
            fig_data['Sales Type'] = fig_data['Sales Type'].map({'predicted_sales': 'Predicted Sales', TARGET_COLUMN: 'Actual Sales'})

            fig = px.line(
                fig_data, x='date', y='Sales', color='Sales Type',
                title=f'Sales Trend for Store {selected_store}',
                labels={'date': 'Date', 'Sales': 'Weekly Sales ($)'},
                template='plotly_white', markers=True
            )
            fig.update_layout(legend_title_text='Legend')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()