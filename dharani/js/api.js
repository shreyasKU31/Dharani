// --- Private Helper Functions ---
const getMean = (arr) => arr.reduce((sum, val) => sum + val, 0) / arr.length;
const formatDate = (date) => date.toISOString().slice(0, 10).replace(/-/g, "");

// --- Public API Functions ---

export async function get5yrAvgWeather(lat, lon) {
  const results = {};
  const fillValue = -999;
  const apiUrl = "https://power.larc.nasa.gov/api/temporal/daily/point";

  const today = new Date();
  const endDate = new Date(today);
  endDate.setDate(today.getDate() - 1);
  const startDate = new Date(endDate);
  startDate.setFullYear(endDate.getFullYear() - 5);

  const params = new URLSearchParams({
    start: formatDate(startDate),
    end: formatDate(endDate),
    latitude: lat,
    longitude: lon,
    community: "RE",
    parameters: "T2M,RH2M,PRECTOTCORR",
    format: "JSON",
  });

  const response = await fetch(`${apiUrl}?${params}`);
  if (!response.ok) throw new Error(`NASA API Error: ${response.statusText}`);
  const data = await response.json();

  const tempAndHumidityData = data.properties.parameter;
  // Return numbers instead of strings
  results.temperature = parseFloat(
    getMean(
      Object.values(tempAndHumidityData.T2M).filter((v) => v !== fillValue)
    ).toFixed(1)
  );
  results.humidity = parseFloat(
    getMean(
      Object.values(tempAndHumidityData.RH2M).filter((v) => v !== fillValue)
    ).toFixed(1)
  );

  const dailyRainData = tempAndHumidityData.PRECTOTCORR;
  const yearlyMonsoonTotals = {};
  for (const dateStr in dailyRainData) {
    const value = dailyRainData[dateStr];
    if (value === fillValue) continue;
    const year = parseInt(dateStr.substring(0, 4), 10);
    const month = parseInt(dateStr.substring(4, 6), 10);
    if (month >= 6 && month <= 9) {
      yearlyMonsoonTotals[year] = (yearlyMonsoonTotals[year] || 0) + value;
    }
  }
  // Return a number instead of a string
  results.rainfall = parseFloat(
    getMean(Object.values(yearlyMonsoonTotals)).toFixed(1)
  );

  return results;
}

export async function getSensorData() {
  try {
    // 1. Wait for the fetch request to complete
    const response = await fetch("http://10.184.166.151:5000/api/data");

    // 2. Check if the request was successful
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // 3. Wait for the JSON data to be parsed from the response
    const data = await response.json();

    // 4. Extract the specific values based on your Pi's JSON output
    const nitrogenValue = data.NPK.nitrogen;
    const phosphorusValue = data.NPK.phosphorus;
    const potassiumValue = data.NPK.potassium;
    const moistureValue = data.SoilMoisture.moisture_percent;

    console.log("Successfully fetched sensor data:", {
      nitrogenValue,
      phosphorusValue,
      potassiumValue,
      moistureValue,
    });

    // 5. Return a clean object with the extracted values
    return {
      nitrogen: nitrogenValue,
      phosphorus: phosphorusValue,
      potassium: potassiumValue,
      moisture: moistureValue,
    };
  } catch (err) {
    console.error("Failed to get sensor data:", err);
    // Re-throw the error so the main function knows something went wrong
    throw err;
  }
}

export async function getAIPrediction(payload) {
  const AI_MODEL_URL =
    "https://asia-south1-dharani-473011.cloudfunctions.net/predict_crop";

  const response = await fetch(AI_MODEL_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`AI Model Error: ${response.statusText}`);
  }

  return response.json();
}
