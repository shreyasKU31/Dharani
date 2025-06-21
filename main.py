from crop_predictor import predict_crop
from tts import speak_message
from stt import get_user_language
from translator import get_translated_message

#predicting crop based on input parameters
n, p, k = 69,51,23
temperature, humidity, ph, rainfall = 22,72,6,106
predicted_crop = predict_crop(n, p, k, temperature, humidity, ph, rainfall)
english_message = f"The predicted crop is {predicted_crop}."
speak_message(english_message)

# Ask for language
speak_message("Which language do you prefer?")

# Geting language from voice
user_language = get_user_language()

# Translateing message and crop
translated_message, lang_code = get_translated_message(english_message, predicted_crop, user_language)

# Speak translated message
speak_message(translated_message, lang=lang_code)
