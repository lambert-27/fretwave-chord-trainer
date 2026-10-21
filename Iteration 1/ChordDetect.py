import librosa
import matplotlib.pyplot as plt
import numpy as np
import queue
import sys
import tempfile
import sounddevice as sd
import soundfile as sf

# 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B, exactly like the 12 pitch classes from a chroma
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
    :param chroma_frame: section of audio as a chroma, to analyse the notes present and find the best
                        chord match
    :return: the best match chord, taken from the average chroma frame
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
        # Make sure the file is opened before starting recording, mode=x ensures that the file can't exist,
        # mode=w however, allowss to overwrite
        with sf.SoundFile(filename, mode="w", samplerate=freq, channels=2,subtype=None) as file:
            # Entering this with block starts the stream, which calls callback for each audio block
            # device=none defaults to systems default
            with sd.InputStream(samplerate=freq, device=None, channels=2, callback=callback):
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