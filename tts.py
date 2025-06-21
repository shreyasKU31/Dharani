from gtts import gTTS
import os
import pygame
import platform
import time

def speak_message(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = "temp_audio.mp3"
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.5)
    pygame.mixer.quit()
    os.remove(filename)
    # Auto-play based on OS
    if platform.system() == "Windows":
        os.system(f'start /min wmplayer "{filename}"')
    elif platform.system() == "Darwin":
        os.system(f'afplay "{filename}"')
    else:
        os.system(f'mpg123 "{filename}"')


