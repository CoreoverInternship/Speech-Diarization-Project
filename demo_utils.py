from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from resemblyzer import sampling_rate
from matplotlib import cm
from time import sleep, perf_counter as timer
from umap import UMAP
from sys import stderr
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr
import wave
from pydub import AudioSegment
from pydub.utils import make_chunks




_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_my_colors = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=float) / 255

# def speech_to_text():
#     recgoniser = sr.Recognizer()

    
#     with sr.AudioFile("audio_data/RENAME_update.wav") as source:
#         audio = recgoniser.record(source)
#     try:
#         text = recgoniser.recognize_google(audio)
#         print("text: "+text)
#     except Exception as e:
#         print("Exception: " + str(e))

def interactive_diarization(similarity_dict, wav, wav_splits, duration, sampling_rate, x_crop=5, show_time=True):
    # sampling_rate = sampling_rate
    # print(sampling_rate)
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], label=name)[0] for name in similarity_dict.keys()]
    text = ax.text(0, 0, "", fontsize=10)
    
    def init():
        ax.set_ylim(0.4, 1)
        ax.set_ylabel("Similarity")
        if show_time:
            ax.set_xlabel("Time (seconds)")
        else:
            ax.set_xticks([])
        ax.set_title("Diarization")
        ax.legend(loc="lower right")
        return lines + [text]
    
    times = [(duration/ len(wav_splits)*s)  for s in range(len(wav_splits))]

    # print(times)
    rate = 1 / (times[1] - times[0])
    crop_range = int(np.round(x_crop * rate))
    ticks = np.arange(0, len(wav_splits), rate)
    ref_time = timer()
    
    def update(i):
        # Crop plot
        crop = (max(i - crop_range // 2, 0), i + crop_range // 2)
        ax.set_xlim(i - crop_range // 2, crop[1])
        if show_time:
            crop_ticks = ticks[(crop[0] <= ticks) * (ticks <= crop[1])]
            ax.set_xticks(crop_ticks)
            ax.set_xticklabels(np.round(crop_ticks / rate).astype(int))

        # Plot the prediction
        similarities = [s[i] for s in similarity_dict.values()]
        best = np.argmax(similarities)
        name, similarity = list(similarity_dict.keys())[best], similarities[best]
        if similarity > 0.75:
            message = "Speaker: %s (confident)" % name
            color = _default_colors[best]
        elif similarity > 0.65:
            message = "Speaker: %s (uncertain)" % name
            color = _default_colors[best]
        else:
            message = "Unknown/No speaker"
            color = "black"
        text.set_text(message)
        text.set_c(color)
        text.set_position((i, 0.96))
        
        # Plot data
        for line, (name, similarities) in zip(lines, similarity_dict.items()):
            line.set_data(range(crop[0], i + 1), similarities[crop[0]:i + 1])
        
        # Block to synchronize with the audio (interval is not reliable)
        current_time = timer() - ref_time
        if current_time < times[i]:
            sleep(times[i] - current_time)
        elif current_time - 0.2 > times[i]:
            print("Animation is delayed further than 200ms!", file=stderr)
        return lines + [text]
    
    ani = FuncAnimation(fig, update, frames=len(wav_splits), init_func=init, blit=not show_time,
                        repeat=False, interval=1)
    # play_wav(wav, blocking=False)
    plt.show()
    
def getSegments(similarity_dict, duration):
        if(not similarity_dict):
            print("No data in similarity dictionary.")
            return

        keys = list(similarity_dict.keys())
        times = [(duration / len(similarity_dict[keys[0]]) * s) for s in range(len(similarity_dict[keys[0]]))]
        tolerance = 0.02
        
        segment_times = {key: [] for key in keys}
        dont_repeat = {key: False for key in keys}
        lastKey =""
        for s in range(len(similarity_dict[keys[0]])):
            for i, key1 in enumerate(keys):
                if dont_repeat[key1]:
                    continue

                is_greater = True
                for j, key2 in enumerate(keys):
                    if i != j and similarity_dict[key1][s] <= similarity_dict[key2][s] + tolerance:
                        is_greater = False
                        break

                if is_greater:
                    segment_times[key1].append(times[s])
                    if(lastKey != ""):
                        segment_times[lastKey].append(times[s])
                    lastKey = key1
                    dont_repeat[key1] = True
                    for other_key in keys:
                        if other_key != key1:
                            dont_repeat[other_key] = False

        for key in keys:
            if(len(segment_times[key]) %2 != 0):
                segment_times[key].append(duration)
           
        return segment_times

def segmentsToText(segment_times,file):
    

    segmentList =[]
    keys = list(segment_times.keys())

    for key in keys:
        
        for i  in range(0,len(segment_times[key]),2):
            
            temp = speech_to_text(segment_times[key][i]-.5,segment_times[key][i+1]+.5,file)
            if(isinstance(temp, str) ):
                text = temp
            else:
                text = "Not Understood"
            start = segment_times[key][i]
            stop = segment_times[key][i+1]
            # print(key+": ",text," start: ",start,"stop: ",stop)
            segmentList.append([key + ": "+text,start,stop])
    return segmentList


def speech_to_text(start,stop,file):
    recgoniser = sr.Recognizer()
    duration = stop - start
    
    with sr.AudioFile(file) as source:
        audio = recgoniser.record(source,duration,start)
    try:
        text = recgoniser.recognize_google(audio)
        return text
    except Exception as e:
        print("Exception: " + str(e))
    


    
        


