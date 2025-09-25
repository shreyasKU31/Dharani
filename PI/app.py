# Save this code as app.py on your Raspberry Pi

# 1. Import necessary libraries
from flask import Flask, jsonify
from flask_cors import CORS  # Import CORS
from datetime import datetime
import random # Used for generating dummy data

# 2. Create the Flask app
app = Flask(__name__)
# IMPORTANT: Enable CORS to allow your webpage to fetch data from this server
CORS(app) 

# 3. --- Mock Sensor Functions ---
# Replace the code in these functions with your actual sensor reading logic.
# For example, using the 'serial' or 'smbus2' library to communicate with your sensors.

def getNPK():
    """
    Reads NPK data from the sensor.
    Returns a dictionary with the values.
    """
    # ---- DUMMY DATA LOGIC (REPLACE THIS) ----
    npk_data = {
        "nitrogen": random.randint(50, 100),    # mg/kg
        "phosphorus": random.randint(20, 60),  # mg/kg
        "potassium": random.randint(20, 60)    # mg/kg
    }
    # ----------------------------------------
    print(f"Read NPK data: {npk_data}")
    return npk_data

def getMoisture():
    """
    Reads soil moisture data from the sensor.
    Returns a dictionary with the percentage.
    """
    # ---- DUMMY DATA LOGIC (REPLACE THIS) ----
    moisture_data = {
        "percentage": round(random.uniform(30.0, 75.0), 1)
    }
    # ----------------------------------------
    print(f"Read Moisture data: {moisture_data}")
    return moisture_data

# 4. Define the API Route
@app.route('/api/getData', methods=['GET'])
def fetch_sensor_data():
    """
    This function is triggered when a GET request is made to /api/getData.
    """
    try:
        # Call your sensor functions to get the latest readings
        npk_values = getNPK()
        moisture_value = getMoisture()

        # Get the current timestamp in UTC
        timestamp = datetime.now(datetime.timeZone.utc).isoformat() + "Z"

        # Structure the data into a single JSON object
        response_data = {
            "timestamp_utc": timestamp,
            "npk": npk_values,
            "soil_moisture": moisture_value
        }
        
        # Use jsonify to properly format the response
        return jsonify(response_data)

    except Exception as e:
        # If any error occurs, return an error message
        error_message = {"error": str(e)}
        print(f"An error occurred: {e}")
        # Return a 500 Internal Server Error status code
        return jsonify(error_message), 500

# 5. Run the Server
if __name__ == '__main__':
    # Use host='0.0.0.0' to make the server accessible on your network
    app.run(host='0.0.0.0', port=6000, debug=True)