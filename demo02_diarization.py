from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *

from pathlib import Path
from multiprocessing import Process
import pygame
import time
from pydub import AudioSegment
import speech_recognition as sr
import wave
from resemblyzer import sampling_rate


# mp3_path = Path("audio_data", "RENAME.mp3")
wav_path = Path("audio_data", "GoodAudio.wav")


# Load the MP3 file
# audio = AudioSegment.from_mp3(mp3_path)

# Export the audio to WAV format
# audio.export(wav_path, format="wav")

with wave.open("audio_data/GoodAudio.wav", "r") as wave_file:
    frames = wave_file.getnframes()
    print("frames: "+str(frames))
    frame_rate = wave_file.getframerate()
    print("frame_rate: "+str(frame_rate))
    duration = frames / float(frame_rate)
    print("duration: "+str(duration))



wav = preprocess_wav(wav_path)



# Cut some segments from single speakers as reference audio
# segments = [[0, 5.5], [6.5, 12], [17, 25]]
# speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
# speaker_wavs = [wav[int(  s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]

segments = [[0,8],[9,17]]
speaker_names = ["quinn", "sarthak"]
speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]
speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]


    
encoder = VoiceEncoder("cpu")
print("Running the continuous embedding on cpu, this might take a while...")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)



speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                zip(speaker_names, speaker_embeds)}




# print(speaker_names)
# print(speaker_embeds)


    
segment_times = getSegments(similarity_dict,duration)

transcript = segmentsToText(segment_times,"audio_data/GoodAudio.wav")
transcript = sorted(transcript,key = lambda x: x[1])

with open('output.txt', 'w') as file:

    for line in transcript:
        text = str(line[0])+ " start:"+str(line[1])+" end:"+str(line[2]) + "\n"
        file.write(text)

# print(speech_to_text(0,6,"audio_data/RENAME.wav"))




# print(wav)
# print(wav_splits)
## Run the interactive demo
#interactive_diarization(similarity_dict, wav, wav_splits, duration, sampling_rate)
# speech_to_text()


