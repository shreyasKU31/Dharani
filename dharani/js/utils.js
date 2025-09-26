/**
 * Tries to get the user's location using GPS first, then falls back to IP geolocation.
 * @returns {Promise<{lat: string, lng: string, source: string}>} A promise resolving with the location.
 */
export async function getLocation() {
  try {
    const position = await new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, {
        timeout: 8000,
      });
    });
    return {
      lat: position.coords.latitude.toFixed(4),
      lng: position.coords.longitude.toFixed(4),
      source: "GPS",
    };
  } catch (gpsError) {
    console.warn(`GPS failed: ${gpsError.message}. Falling back to IP.`);
    try {
      const response = await fetch("https://ipapi.co/json/");
      const data = await response.json();
      return {
        lat: data.latitude.toFixed(4),
        lng: data.longitude.toFixed(4),
        source: "IP Geolocation",
      };
    } catch (ipError) {
      throw new Error("All location methods failed.");
    }
  }
}

/**
 * Determines the current crop season based on the month.
 * @returns {string} The name of the current season.
 */
export function getCurrentCropSeason() {
  const currentMonth = new Date().getMonth() + 1; // 1-12
  if (currentMonth >= 6 && currentMonth <= 10) return "Kharif"; // Monsoon
  if (currentMonth >= 11 || currentMonth <= 3) return "Rabi"; // Winter
  return "Annual";
}
