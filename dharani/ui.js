/**
 * Selects an element from the DOM.
 * @param {string} selector - The CSS selector for the element.
 * @returns {HTMLElement} The found element.
 */
const $ = (selector) => document.querySelector(selector);

/**
 * Updates the display with all the fetched and static data.
 * @param {object} data - The final data object.
 */
export function updateDisplay(data) {
  $("#n-value").textContent = data.n;
  $("#p-value").textContent = data.p;
  $("#k-value").textContent = data.k;
  $("#ph-value").textContent = data.pH;
  $("#humidity-value").textContent = `${data.humidity}%`;
  $("#temp-value").textContent = `${data.temperature}°C`;
  $("#rainfall-value").textContent = `${data.rainfall} mm`;
  $("#season-value").textContent = data.season;
}

/**
 * Updates the location information on the page.
 * @param {object} location - The location object { lat, lng, source }.
 */
export function updateLocationDisplay(location) {
  $("#lat").textContent = location.lat;
  $("#lng").textContent = location.lng;
  $("#source").textContent = location.source;
}

/**
 * Updates the status message shown to the user.
 * @param {string} message - The message to display.
 */
export function updateStatus(message) {
  $("#status").textContent = message;
}

/**
 * Controls the state of the main button.
 * @param {boolean} isLoading - True to disable the button, false to enable it.
 */
export function setButtonLoading(isLoading) {
  const button = $("#receiveDataBtn");
  button.disabled = isLoading;
  button.textContent = isLoading ? "Loading..." : "Get Latest Data";
}
