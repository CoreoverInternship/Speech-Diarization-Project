from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from multiprocessing import Process, Event
import pygame
import time
from pydub import AudioSegment
import speech_recognition as sr


# def main():
#     playaudio()

def playaudio(start_event):
    pygame.mixer.init(frequency = 55125)
    path = Path("audio_data", "demo_big.mp3")
    pygame.mixer.music.load(path)
    start_event.wait()  
    pygame.mixer.music.play(start=0)
    while pygame.mixer.music.get_busy():  # Check if the music is still playing
        time.sleep(1)  # Wait for the music to finish



def code_run():
    # start_event.wait()

    ## Get reference audios
    recgoniser = sr.Recognizer()
    mp3_path = Path("audio_data", "demo_big.mp3")
    wav_path = Path("audio_data", "demo_updated.wav")
    print("got here")
    with sr.AudioFile("audio_data/demo_updated.wav") as source:
        audio = recgoniser.record(source)
    try:
        text = recgoniser.recognize_google(audio)
        print("text: "+text)
    except Exception as e:
        print("Exception: " + str(e))

    mp3_audio = AudioSegment.from_file(mp3_path, format="mp3")
    mp3_audio.export(wav_path, format="wav")

    # wav_fpath = Path("audio_data", "demo_big.mp3")
    wav = preprocess_wav(wav_path)

    # Cut some segments from single speakers as reference audio
    # segments = [[0, 5.5], [6.5, 12], [17, 25]]
    segments = [[2,17],[19,42]]
    # speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
    speaker_names = ["quinn", "sarthak"]
    # speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]
    speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]

    
        
    encoder = VoiceEncoder("cpu")
    print("Running the continuous embedding on cpu, this might take a while...")
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)



    speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
    similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                    zip(speaker_names, speaker_embeds)}
    print(speaker_names)
    print(speaker_embeds)
    # print(similarity_dict)
    # print(wav)
    # print(wav_splits)
    ## Run the interactive demo
    interactive_diarization(similarity_dict, wav, wav_splits)
 
    
if __name__ == "__main__":
    # start_event = Event()

    # playaudio()
    code_run()
   
    # music_process = multiprocessing.Process(target=playaudio)

    # task_process = multiprocessing.Process(target=code_run)

    # music_process = Process(target=playaudio, args=(start_event,))
    # task_process = Process(target=code_run, args=(start_event,))


    # task_process.start()
    
    # music_process.start()
    # # music_process.sleep(1)
    
    # time.sleep(2)  # Allow some setup time if necessary before starting both processes
    # start_event.set()  # 


    # music_process.join()
    # task_process.join()