"""
Purpose: Functions extracted from First Iteration notebooks to perform chord and onset detection
Author: Mark Lambert

Provides a bank of chord templates represented as 12 length vectors mapped to each pitch class
"""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import queue
import sys
import tempfile
import sounddevice as sd
import soundfile as sf


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
    
    :return: A string that labels each sequence of chord detected and the score

    """
    result = ""

    onset_frames = librosa.onset.onset_detect(y=chords, sr=sr)
    chroma = librosa.feature.chroma_stft(y=chords, sr=sr)
    
    for i in range (1, len(onset_frames)):
        prev = onset_frames[i-1]
        cur = onset_frames[i]
    
        # using multi demnsional array slicing, we keep all 12 rows (the pitch class)
        # and slice from frame prev to frame cur
        strum_slice = chroma[:, prev: cur]
    
        # calculate the average chroma for the slice
        avg_chroma = np.mean(strum_slice, axis=1)
    
        chord, score = detect_chord(avg_chroma)
        print(f"Chord: {chord}, Score: {score}")
        
        result += f"{chord} {score:.2f} " 
    return result
    
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
    clip, samplerate = librosa.load('clips/testclip.wav', sr=None)
    chroma = librosa.feature.chroma_cqt(y=clip, sr=samplerate)
    avg_chroma = np.mean(chroma, axis=1)
    return detect_chord(avg_chroma)