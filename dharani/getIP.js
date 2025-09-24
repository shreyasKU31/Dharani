// --- Private Helper Functions for finding location ---

function tryGPS() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      return reject("GPS not supported by this browser.");
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        resolve({
          lat: position.coords.latitude.toFixed(4),
          lng: position.coords.longitude.toFixed(4),
          source: "GPS",
        });
      },
      (error) => reject(`GPS failed: ${error.message}`),
      { enableHighAccuracy: false, timeout: 10000, maximumAge: 300000 }
    );
  });
}

function tryIPAPI(url, sourceName, latKey, lonKey) {
  return new Promise((resolve, reject) => {
    fetch(url)
      .then((response) => {
        if (!response.ok)
          throw new Error(`HTTP error! Status: ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (data[latKey] && data[lonKey]) {
          resolve({
            lat: Number(data[latKey]).toFixed(4),
            lng: Number(data[lonKey]).toFixed(4),
            source: `IP (${sourceName})`,
          });
        } else {
          reject(`IP API (${sourceName}) failed: Invalid data received.`);
        }
      })
      .catch((error) =>
        reject(`IP API (${sourceName}) error: ${error.message}`)
      );
  });
}

// --- Public Function ---

/**
 * Tries to get the user's location using GPS first, then falls back to IP geolocation.
 * @returns {Promise<{lat: string, lng: string, source: string}>} A promise that resolves with the location object.
 */
export async function getLocation() {
  const methods = [
    () => tryGPS(),
    () =>
      tryIPAPI("https://ipapi.co/json/", "ipapi.co", "latitude", "longitude"),
    () =>
      tryIPAPI(
        "http://ip-api.com/json/?fields=lat,lon",
        "ip-api.com",
        "lat",
        "lon"
      ),
    () =>
      tryIPAPI(
        "https://freeipapi.com/api/json",
        "freeipapi.com",
        "latitude",
        "longitude"
      ),
  ];

  for (const method of methods) {
    try {
      const result = await method();
      console.log(`Location found via ${result.source}`);
      return result; // Success! Return the first result we get.
    } catch (error) {
      console.warn(error); // Log the warning and try the next method.
    }
  }

  // If all methods failed
  throw new Error("All location methods failed.");
}
