import pandas as pd
import uvicorn
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.predict import make_predictions, model as loaded_model



# --- 1. APP INITIALIZATION ---
app = FastAPI(
    title="Store Sales Prediction API",
    description="FastAPI backend to predict weekly sales for stores based on uploaded data.",
)

# --- 2. API ENDPOINTS ---

@app.get("/", tags=["Health Check"])
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"status": "API is running"}

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to receive a CSV or XLSX file, make predictions, and return them as JSON.
    
    Args:
        file (UploadFile): The data file for prediction.
                           Must be a .csv or .xlsx file.
                           
    Returns:
        JSONResponse: A JSON object containing the predictions or an error message.
    """
    if loaded_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not available. Please ensure the model is trained and loaded."
        )

    filename = file.filename
    
    try:
        contents = await file.read()
        
        # --- Load data into DataFrame based on the file extension ---
        if filename.endswith('.csv'):
            input_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif filename.endswith('.xlsx'):
            input_df = pd.read_excel(io.BytesIO(contents))

        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV or XLSX file.")
            
        print(f"Received data from '{filename}' for prediction with {len(input_df)} rows.")

        predictions_df = make_predictions(input_df)
        
        if predictions_df is None:
             raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")

        # --- Format response ---
        predictions_json = predictions_df.to_dict(orient='records')
        
        return JSONResponse(content={
            "message": "Predictions generated successfully.",
            "predictions": predictions_json
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")


if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)