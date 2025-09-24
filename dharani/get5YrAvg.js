/**
 * Calculates the mean of an array of numbers.
 * @param {number[]} arr - The array of numbers.
 * @returns {number} The mean of the array.
 */
const getMean = (arr) => arr.reduce((sum, val) => sum + val, 0) / arr.length;

/**
 * Formats a Date object into a YYYYMMDD string for the NASA API.
 * @param {Date} date - The date to format.
 * @returns {string} The formatted date string.
 */
const formatDate = (date) => date.toISOString().slice(0, 10).replace(/-/g, "");

/**
 * Provides a comprehensive 5-year weather summary using NASA POWER data.
 * @param {number} lat - The latitude of the location.
 * @param {number} lon - The longitude of the location.
 * @returns {Promise<object>} A promise that resolves to an object with weather values.
 */
export async function get5yrAvgWeather(lat, lon) {
  const results = {};
  const fillValue = -999;
  const apiUrl = "https://power.larc.nasa.gov/api/temporal/daily/point";

  // --- PART 1: Calculate 5-Year Average Temp & Humidity ---
  const today = new Date();
  const endDate5yr = new Date(today);
  endDate5yr.setDate(today.getDate() - 1); // Data is available up to yesterday

  const startDate5yr = new Date(endDate5yr);
  startDate5yr.setFullYear(endDate5yr.getFullYear() - 5);

  const params5yr = new URLSearchParams({
    start: formatDate(startDate5yr),
    end: formatDate(endDate5yr),
    latitude: lat,
    longitude: lon,
    community: "RE",
    parameters: "T2M,RH2M,PRECTOTCORR", // Temp, Humidity, and Rainfall
    format: "JSON",
  });

  const response5yr = await fetch(`${apiUrl}?${params5yr}`);
  if (!response5yr.ok)
    throw new Error(`NASA API Error: ${response5yr.statusText}`);
  const data5yr = await response5yr.json();

  // Process Temp and Humidity
  for (const [paramCode, name] of Object.entries({
    T2M: "temperature",
    RH2M: "humidity",
  })) {
    const dailyData = data5yr.properties.parameter[paramCode];
    const validValues = Object.values(dailyData).filter((v) => v !== fillValue);
    results[name] =
      validValues.length > 0 ? getMean(validValues).toFixed(1) : "N/A";
  }

  // --- PART 2: Calculate 5-Year Average of Total Monsoon Rainfall ---
  const dailyRainData = data5yr.properties.parameter.PRECTOTCORR;
  const yearlyMonsoonTotals = {};

  for (const dateStr in dailyRainData) {
    const value = dailyRainData[dateStr];
    if (value === fillValue) continue;

    const year = parseInt(dateStr.substring(0, 4), 10);
    const month = parseInt(dateStr.substring(4, 6), 10);

    // Monsoon season in India is typically June (6) to September (9)
    if (month >= 6 && month <= 9) {
      if (!yearlyMonsoonTotals[year]) {
        yearlyMonsoonTotals[year] = 0;
      }
      yearlyMonsoonTotals[year] += value;
    }
  }

  const validYearlyTotals = Object.values(yearlyMonsoonTotals);
  results.rainfall =
    validYearlyTotals.length > 0
      ? getMean(validYearlyTotals).toFixed(1)
      : "N/A";

  return results;
}
