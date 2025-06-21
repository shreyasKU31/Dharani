import speech_recognition as sr

def get_user_language():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for language...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("Recognized language:", text)
        return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return "english"
    except sr.RequestError:
        print("Could not request results")
        return "english"
