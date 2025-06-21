import pandas as pd
import joblib

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    model = joblib.load('crop_model.pkl')
    input_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    return model.predict(input_data)[0]
