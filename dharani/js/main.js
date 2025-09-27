import * as api from "./api.js";
import * as ui from "./ui.js";
import * as utils from "./utils.js";
import { speakInKannada } from "./speech.js";

async function handleGetPrediction() {
  try {
    ui.setButtonLoading(true);

    // 1. Get Location and Season
    ui.updateStatus("Detecting your location...");
    const location = await utils.getLocation();
    ui.updateLocationDisplay(location);
    const season = utils.getCurrentCropSeason();

    // 2. Fetch Data from all sources concurrently
    ui.updateStatus("Gathering sensor and weather data...");
    const [sensorData, weatherData] = await Promise.all([
      api.getSensorData(),
      api.get5yrAvgWeather(location.lat, location.lng),
    ]);

    // 3. Prepare data for display
    const displayData = {
      n: sensorData.nitrogen,
      p: sensorData.phosphorus,
      k: sensorData.potassium,
      moisture: sensorData.moisture,
      pH: 7.0, // Using default pH of 7.0
      temperature: weatherData.temperature,
      humidity: weatherData.humidity,
      rainfall: weatherData.rainfall,
      season: season,
    };

    console.log(displayData);
    ui.updateDisplay(displayData);

    // 4. Send data to the AI Model in the required format
    ui.updateStatus("Asking AI model for prediction...");

    // This object's keys now EXACTLY match your requirements
    // const modelPayload = {
    //   Season: displayData.season,
    //   "N (kg/ha)": displayData.n,
    //   "P (kg/ha)": displayData.p,
    //   "K (kg/ha)": displayData.k,
    //   pH: 7,
    //   "Moisture (%)": displayData.moisture,
    //   "Temp (°C)": displayData.temperature,
    //   "Rainfall (mm)": displayData.rainfall,
    //   "Humidity (%)": displayData.humidity,
    // };
    const modelPayload = {
      Season: "Kharif",
      "N (kg/ha)": 145,
      "P (kg/ha)": 70,
      "K (kg/ha)": 69,
      pH: 7.5,
      "Moisture (%)": 77,
      "Temp (°C)": 33,
      "Rainfall (mm)": 966,
      "Humidity (%)": 65,
    };

    console.log(modelPayload);

    const predictionResult = await api.getAIPrediction(modelPayload);

    const cropName = predictionResult.best_crop;
    ui.updatePrediction(cropName); // Update the text on the screen
    speakInKannada(cropName);

    ui.updateStatus("Success! Prediction complete.");
  } catch (error) {
    console.error("An error occurred:", error);
    ui.updateStatus(`Error: ${error.message}`);
    ui.updatePrediction("Failed");
  } finally {
    ui.setButtonLoading(false);
  }
}

document
  .getElementById("receiveDataBtn")
  .addEventListener("click", handleGetPrediction);
