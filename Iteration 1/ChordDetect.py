"""
Purpose: Functions extracted from First Iteration notebooks to record and perform chord and onset detection
Provides a bank of chord templates represented as 12 length vectors mapped to each pitch class 

Author: Mark Lambert

"""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import queue
import sys
import tempfile
import sounddevice as sd
import soundfile as sf
import time


#dictionary of chord templates, 
# 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B, 
# exactly like the 12 pitch classes from a chroma
chord_templates = {
    'C':  [1,0,0,0,1,0,0,1,0,0,0,0],
    'G':  [0,0,1,0,0,0,0,1,0,0,0,1],
    'D':  [0,0,1,0,0,0,1,0,0,1,0,0],
    'Dm': [0,0,1,0,0,1,0,0,0,1,0,0],
    'E':  [0,0,0,0,1,0,0,0,1,0,0,1],
    'Em': [0,0,0,0,1,0,0,1,0,0,0,1],
    'Am': [1,0,0,0,1,0,0,0,0,1,0,0],
    'F':  [1,0,0,0,0,1,0,0,0,1,0,0],
    'A':  [0,1,0,0,1,0,0,0,0,1,0,0],
}

def detect_chord(chroma_frame):
    '''
    Detect notes in a chroma vector using cosine similarity against the 
        template chord bank and return the closest match
    
    :param chroma_frame: the 12 element average chroma vector
    
    :return: A tuple containing the best match chord, taken from the average chroma frame and the certainty score of the algorithm
    '''

    best_match = None
    best_score = -1

    for name, template in chord_templates.items():
        #Cosine similarity: dot product of the the chroma_frame vector and the template of chords vector divided by the product of their magnitudes
        score = np.dot(chroma_frame, template) / (np.linalg.norm(chroma_frame) * np.linalg.norm(template))
        if score > best_score:
            best_match = name
            best_score = score

    return best_match, best_score
    
def get_chords(chords, sr):
    """
    Divides an audio clip into individual onsets and prints the detected chords for each onset segment (prev and cur)
    
    :param chords: the raw audio loaded by librosa.load(), stored as a numpy array
    :param sr: The sample rate of the audio track
    
    :return: A list of tuples containing the detected chords and percentage similarity score of the algorithm

    """
    onset_frames = librosa.onset.onset_detect(y=chords, sr=sr)
    # swapped from stft to cqt. cqt uses logarithmic frequency spacing 
    # to map to each semitone (12 pitch classes) gives much higher accuracy
    chroma = librosa.feature.chroma_cqt(y=chords, sr=sr)
    # append the final frame so the segment after the last onset is also analysed
    onset_frames = np.append(onset_frames, chroma.shape[1])
    # a list starting at first strum frame, will hold only relevant
    # onsets w/ noise discarded
    strum_group = [onset_frames[0]]
    detected = []
    buffer = 0.5
    # iterate each raw onset, filter out any that happen within buffer length of time
    for i in range (1, len(onset_frames)):
        prev = onset_frames[i-1]
        cur = onset_frames[i]

        frame_since_last = cur - prev
        if librosa.frames_to_time(frame_since_last, sr=sr) > buffer:
            strum_group.append(cur)

    # then iterate through the strumgroup list and define an end for each slice of chord
    for index, strum_start in enumerate(strum_group):
        if index + 1 < len(strum_group):
            strum_end = strum_group[index+1]
        else:
            strum_end = len(chroma[0])               
        if strum_start >= strum_end:
            continue
        # using multi demnsional array slicing, we keep all 12 rows (the pitch class)
        # and slice from frame prev to frame cur
        strum_slice = chroma[:, strum_start: strum_end]
    
        # calculate the average chroma for the slice
        avg_chroma = np.mean(strum_slice, axis=1)
        
        chord, score = detect_chord(avg_chroma)
        percent = score*100
        print(f"Chord: {chord}, Score:{percent:.2f}%")
        detected.append((chord, percent))
    return detected
    
def check_chords(detected, prompted):
    """
    Compares each detected chord against the chord the user was prompted with, in the order they appeared when printed
        
    The function looksahead by one to the next chord in the event that the user makes a mistake allowing them one margin of error.
    This helps with the current sensitivity of the onsets that was making the notebook very challenging for a newcomer.
        

    :param detected: list of tuples containing (chord, percent) returned from get_chords()
    :param prompted: list of chord names the user was prompted with, in order they appeared when printed

    :return: list of tuples, true if the chord occured at the same position as the detected along with the prompted chord name
    """
    results = []
    
    for i, chord in enumerate(prompted):
        is_correct = False
        # Check if a chord was detected at this index and if it matches the target
        if i < len(detected) and detected[i][0] == chord:
            is_correct = True
        elif i+1 < len(detected) and detected[i+1][0] == chord:
            is_correct = True
            i += 1
        result = (chord, is_correct)
        results.append(result)
    return results

def record():
    """
    Microphone input is continuously recorded by writing blocks directly to a file

    Uses a producer-consumer model with a queue as the store. Recording stops once the user terminates the kernel
        
    :return: The loaded audio clip and samplerate of the track

    """
    # producer-consumer model for continuous recording. Using a queue and a callback to 
    # produce blocks of audio that are pushed to the queue and then pulled 
    # by the q.get(), to consume from the buffer and written to the file
    filename = "clips/testclip.wav" 
    freq = 44100  
    q = queue.Queue()

    # added a small sleep to allow for getting ready to strum
    time.sleep(1)
    def callback(indata, _, __, ___):
        """This is called from a separate thread for each audio block."""
        q.put(indata.copy())
    
    try:
        # bug fix: use query_devices to get the systems default input device channel count
        channels = int(sd.query_devices(device=None, kind='input')['max_input_channels'])
        # Make sure the file is opened before starting recording, mode=x ensures that the file can't exist,
        # mode=w however, allowss to overwrite. 
        with sf.SoundFile(filename, mode="w", samplerate=freq, channels=channels,subtype=None) as file:
            # Entering this with block starts the stream, which calls callback for each audio block
            # device=none defaults to systems default
            with sd.InputStream(samplerate=freq, device=None, channels=channels, callback=callback):
                print("Recording, press the stop button in Jupyter Notebook to stop")
                while True:
                    file.write(q.get())
    
    except KeyboardInterrupt:
        print("\nRecording finished: " + repr(filename))
    except Exception as e:
        print(f"Error: {type(e).__name__} - {e}")

    clip, sr = librosa.load('clips/testclip.wav', sr=None)
    return clip, sr

def analyse():
    """
        Helper function used during notebook research that loads the clip saved by record(),
        calculate the average chroma and detect the chord

        :return: The detected chord and similarity score from detect_chord
    """
    clip, samplerate = librosa.load('clips/testclip.wav', sr=None)
    chroma = librosa.feature.chroma_cqt(y=clip, sr=samplerate)
    avg_chroma = np.mean(chroma, axis=1)
    return detect_chord(avg_chroma)