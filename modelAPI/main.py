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
    """HTTP Cloud Function to predict the best crop."""
    headers = {"Access-Control-Allow-Origin": "*"}

    if request.method == "OPTIONS":
        return ("", 204, headers)

    request_json = request.get_json(silent=True)
    if not request_json:
        return ({"error": "Invalid JSON"}, 400, headers)
    
    try:
        # Ensure input order matches feature names
        input_data = pd.DataFrame([request_json], columns=feature_names)

        # Encode Season if encoder is available
        if season_encoder:
            input_data['Season'] = season_encoder.transform(input_data['Season'])

        # Make prediction
        prediction = model.predict(input_data)

        # Decode crop name if crop_names exist
        best_crop = crop_names[prediction[0]] if crop_names is not None else prediction[0]

        result = {"best_crop": best_crop}
        return (result, 200, headers)
        
    except Exception as e:
        return ({"error": str(e)}, 500, headers)
