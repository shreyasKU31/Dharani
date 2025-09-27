// The URL of the NEW Cloud Function you just deployed.
// ⚠️ You'll need to replace this with your actual URL after deployment.
const TTS_FUNCTION_URL =
  "https://asia-south1-dharani-473011.cloudfunctions.net/getKannadSpeech";

/**
 * Fetches synthesized Kannada speech from the cloud and plays it.
 * @param {string} text - The text to be spoken.
 */
export async function speakInKannada(text) {
  try {
    // 1. Send the text to our cloud function
    const response = await fetch(TTS_FUNCTION_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });

    if (!response.ok) {
      throw new Error(`TTS service failed: ${response.statusText}`);
    }

    const data = await response.json();

    // 2. Create an audio source from the Base64 response and play it
    const audioSource = `data:audio/mp3;base64,${data.audioContent}`;
    const audio = new Audio(audioSource);
    audio.play();
  } catch (error) {
    console.error("Failed to play speech:", error);
  }
}
