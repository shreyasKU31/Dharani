const $ = (selector) => document.querySelector(selector);

export function updateDisplay(data) {
  $("#n-value").textContent = data.n;
  $("#p-value").textContent = data.p;
  $("#k-value").textContent = data.k;
  $("#ph-value").textContent = data.pH;
  $("#humidity-value").textContent = `${data.humidity}%`; // Add unit
  $("#temp-value").textContent = `${data.temperature}°C`; // Add unit
  $("#rainfall-value").textContent = `${data.rainfall} mm`; // Add unit
  $("#season-value").textContent = data.season;
}

export function updateLocationDisplay(location) {
  $("#lat").textContent = location.lat;
  $("#lng").textContent = location.lng;
  $("#source").textContent = location.source;
}

export function updateStatus(message) {
  $("#status").textContent = message;
}

export function updatePrediction(cropName) {
  $("#crop-prediction").textContent = cropName;
}

export function setButtonLoading(isLoading) {
  const button = $("#receiveDataBtn");
  button.disabled = isLoading;
  button.textContent = isLoading ? "Loading..." : "Get Crop Prediction";
}
