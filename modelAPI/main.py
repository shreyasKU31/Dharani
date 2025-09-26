import functions_framework
import joblib
import pandas as pd
from google.cloud import storage

# --- Global variables (loaded once per function instance) ---
MODEL_BUCKET = "dharani_model"
MODEL_FILE = "dharani.joblib"
LOCAL_MODEL_PATH = f"/tmp/{MODEL_FILE}"

# Download and load the model bundle from Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(MODEL_BUCKET)
blob = bucket.blob(MODEL_FILE)
blob.download_to_filename(LOCAL_MODEL_PATH)

# Load the dictionary bundle
model_bundle = joblib.load(LOCAL_MODEL_PATH)

# Extract classifier and metadata
model = model_bundle['model']
feature_names = model_bundle['feature_names']
crop_names = model_bundle['crop_names']
season_encoder = model_bundle.get('season_encoder')  # Optional, if used

print("✅ Model loaded successfully!")

@functions_framework.http
def predict_crop(request):
    """HTTP Cloud Function to predict the best crop with CORS handling."""

    # This handles the browser's preflight OPTIONS request for CORS
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return ("", 204, headers)

    # These headers are for the main POST request
    headers = {"Access-Control-Allow-Origin": "*"}

    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return ({"error": "Invalid JSON"}, 400, headers)

        input_data = pd.DataFrame([request_json], columns=feature_names)

        if season_encoder:
            input_data['Season'] = season_encoder.transform(input_data['Season'])

        prediction = model.predict(input_data)
        best_crop = crop_names[prediction[0]] if crop_names is not None else prediction[0]

        return ({"best_crop": best_crop}, 200, headers)

    except Exception as e:
        return ({"error": str(e)}, 500, headers)
