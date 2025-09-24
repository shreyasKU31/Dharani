import { getLocation } from "./getIP.js";
import { get5yrAvgWeather } from "./get5YrAvg.js";
import * as ui from "./ui.js";

/**
 * Determines the current Indian crop season based on the month.
 * @returns {string} The name of the crop season.
 */
function getCurrentCropSeason() {
  const currentMonth = new Date().getMonth() + 1; // 1-12
  if (currentMonth >= 6 && currentMonth <= 10) return "Kharif"; // Monsoon season
  if (currentMonth >= 11 || currentMonth <= 3) return "Rabi"; // Winter season
  return "Zaid"; // Summer season (April, May)
}

/**
 * Main function to handle the data fetching and display process.
 */
async function handleReceiveData() {
  try {
    ui.setButtonLoading(true);

    // 1. Get Location
    ui.updateStatus("Detecting your location...");
    const location = await getLocation();
    ui.updateLocationDisplay(location);

    // 2. Fetch Weather Data using the location
    ui.updateStatus("Fetching 5-year weather averages from NASA...");
    const weather = await get5yrAvgWeather(8.5241, 76.9366);

    // 3. Combine all data
    const finalData = {
      // Static sensor data (as in your example)
      n: 83,
      p: 53,
      k: 52,
      pH: 4.6,
      // Dynamic data
      temperature: weather.temperature,
      humidity: weather.humidity,
      rainfall: weather.rainfall,
      season: getCurrentCropSeason(),
    };

    // 4. Update the UI with final data
    ui.updateDisplay(finalData);
    ui.updateStatus("Success! All data has been updated.");
  } catch (error) {
    console.error("An error occurred:", error);
    ui.updateStatus(`Error: ${error.message}`);
  } finally {
    // 5. Always re-enable the button
    ui.setButtonLoading(false);
  }
}

// Attach the main function to the button's click event.
document
  .getElementById("receiveDataBtn")
  .addEventListener("click", handleReceiveData);
