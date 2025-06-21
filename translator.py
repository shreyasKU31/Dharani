from crop_translations import crop_translations

# Language mapping (spoken name to ISO code)
language_map = {
    "kannada": "kn",
    "telugu": "te",
    "hindi": "hi",
    "malayalam": "ml",
    "tamil": "ta"
}

def get_translated_message(message, crop, user_lang):
    lang_code = language_map.get(user_lang.lower())

    if not lang_code:
        return message, 'en'  # default to English if unsupported

    translated_crop = crop_translations.get(crop.lower(), {}).get(lang_code, crop)

    # Custom translated message based on language code
    translated_message = {
        "kn": f"ನಿಮ್ಮ ಭೂಮಿಗೆ ಅನುಕೂಲವಾಗಿರುವ ಬೆಳೆ {translated_crop} ಆಗಿದೆ.",
        "hi": f"आपकी भूमि के लिए उपयुक्त फसल {translated_crop} है।",
        "ta": f"உங்கள் நிலத்திற்கு ஏற்ப பரிந்துரைக்கப்படும் பயிர் {translated_crop} ஆகும்.",
        "te": f"మీ భూమికి సరైన పంట {translated_crop} గా అంచనా వేయబడింది.",
        "ml": f"നിങ്ങളുടെ ഭൂമിക്ക് അനുയോജ്യമായ വിള {translated_crop} ആണെന്ന് നിർണ്ണയിച്ചിട്ടുണ്ട്."
    }.get(lang_code, message)

    return translated_message, lang_code
