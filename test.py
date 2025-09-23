import requests

url = "https://asia-south1-dharani-473011.cloudfunctions.net/predict_crop"
payload = {
    'Season': 'Kharif',
        'N (kg/ha)': 53,
        'P (kg/ha)': 42,
        'K (kg/ha)': 28,
        'pH': 6.0,
        'Moisture (%)': 60,
        'Temp (°C)': 26,
        'Rainfall (mm)': 970,
        'Humidity (%)': 76
}



response = requests.post(url, json=payload)
print(response.json())
