import functions_framework
import joblib
import pandas as pd
from google.cloud import storage

# --- Global variables (loaded once per function instance) ---
# ⚠️ Make sure to change this to your actual bucket name!
MODEL_BUCKET = "YOUR_BUCKET_NAME_HERE"
MODEL_FILE = "model.joblib"
LOCAL_MODEL_PATH = f"/tmp/{MODEL_FILE}"

# Download and load the model from Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(MODEL_BUCKET)
blob = bucket.blob(MODEL_FILE)
blob.download_to_filename(LOCAL_MODEL_PATH)
model = joblib.load(LOCAL_MODEL_PATH)
print("✅ Model loaded successfully!")

@functions_framework.http
def predict_crop(request):
    """HTTP Cloud Function to predict the best crop."""
    headers = {"Access-Control-Allow-Origin": "*"}

    if request.method == "OPTIONS":
        return ("", 204, headers)

    request_json = request.get_json(silent=True)
    if not request_json:
        return ({"error": "Invalid JSON"}, 400, headers)
    
    try:
        # The exact feature order your model was trained on
        feature_order = ['N', 'P', 'K', 'ph', 'soil_moisture', 'latitude', 'longitude']
        
        input_data = pd.DataFrame([request_json], columns=feature_order)
        prediction = model.predict(input_data)
        result = {"best_crop": prediction[0]}
        
        return (result, 200, headers)
        
    except Exception as e:
        return ({"error": str(e)}, 500, headers)