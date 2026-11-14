"""
Purpose: Functions extracted from First Iteration notebooks to record and perform chord and onset detection
Provides a bank of chord templates represented as 12 length vectors mapped to each pitch class 

Author: Mark Lambert

"""
import librosa
import numpy as np
import queue
import sys
import sounddevice as sd
import soundfile as sf
import time


# 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B, 
maj_template = [1,0,0,0,1,0,0,1,0,0,0,0]
min_template = [1,0,0,1,0,0,0,1,0,0,0,0]
chord_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
chord_templates = {}

for name in chord_names:
    chord_templates[name] = maj_template
    chord_templates[name+'m'] = min_template
    # list slicing, removes the last element in the list and appends it to the front 
    maj_template = [maj_template[-1]] + maj_template[0:-1]
    min_template = [min_template[-1]] + min_template[0:-1]
    
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
    

def get_onsets(clip, sr): 
    '''
        Detects onsets of a clip and divides them into tuples containing (start, end) frames for 
            groups of strums and transitions

        :param clip: the raw audio loaded by librosa.load(), stored as a numpy array
        :param sr: The sample rate of the audio track

        :return strum_group: A list of tuples, each marking the beginning and ending of a chord (start, end)
        :return transition_group: A list of tuples, each marking the end of a chord and the beginning of the next (start, end)
    '''
    # basic normalisation so that the rms becomes more accurate
    clip = librosa.util.normalize(clip)
    rms = librosa.feature.rms(y=clip)[0]
    onset_frames = librosa.onset.onset_detect(y=clip, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=clip, sr=sr)
    onset_frames = np.append(onset_frames, chroma.shape[1])
    
    valid_threshold = 0.2 * rms.max()
    frame_offset = 2
    buffer = 0.5

    # lists of tuples w/ (start, end) frames
    strum_group = []
    transition_group = []

    # list of valid strum onsets detected by rms
    strums = []

    # find strum end onsets
    for i in range(len(onset_frames)):
        frame = onset_frames[i]

        # only check for strum end onsets if there has already been a detected onset
        if i > 0:
            prev = onset_frames[i-1]
            cur = onset_frames[i]
    
            frame_since_last = cur - prev
            if librosa.frames_to_time(frame_since_last, sr=sr) > buffer:
                strums.append(cur)
                continue
        
        # find starts w/ rms
        if frame < len(rms):
            curr = rms[frame]
            prev = rms[frame-frame_offset]
        
        # check if above valid threshold (filter noise) and
        # curr is louder than preevious, showing rise in energy
        if curr > valid_threshold and curr > 1.4 * prev:
            strums.append(frame)

    # sort the strum list so that they are in chronological order
    strums = sorted(set(strums))

    # then iterate through each set of onsets
    for i in range(len(strums) - 1):
        start = strums[i]
        end = strums[i+1]

        if start >= end:
            continue

        # rms always calculates the first strum, so 
        # i mod 2 = 0 means we are on even index (0, 2, 4) 
        # and all even indices are strum segments
        if i % 2 == 0:
            strum_group.append((start, end))
        # otherwise the mod result is 1, so any odd
        # index is a transition segment
        else:
            transition_group.append((start, end))

    # return the strum and transition segments as separate lists of tuples (start, end)
    return strum_group, transition_group

def detect_chords(strum_group, chroma):
    '''
        Detects each chord in the list of tuples containing onsets for valid strums

        :param strum_group: a list of tuples, containing onsets that mark the start and end of chords (start, end)
        :param chroma: the chroma vector computed for the source clip

        :return detected: a list of tuples containing the chord name and similarity score of the algorithm
    '''
    detected = []
    # then iterate through the strumgroup list and define an end for each slice of chord
    for strum_start, strum_end in strum_group:
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

def get_chords(clip, sr):
    '''
        Helper function used by the chord_prompt notebooks that act as the gateway to triggering the 
            rest of the chord and onset detection logic

        :param clip: the raw audio loaded by librosa.load(), stored as a numpy array
        :param sr: The sample rate of the audio track

        :return detected: a list of tuples containing the chord name and similarity score of the algorithm
    '''
    chroma = librosa.feature.chroma_cqt(y=clip, sr=sr)
    strum_group, transition_group = get_onsets(clip, sr)
    # TODO detect_transition_times()?
    return detect_chords(strum_group, chroma)
    
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
    # gives the user a margin of error
    lookahead = 0
    
    for i, chord in enumerate(prompted):
        is_correct = False
        # index variable, for lookahead
        index = i + lookahead 
        # Check if a chord was detected at this index and if it matches the target
        if index < len(detected) and detected[index][0] == chord:
            is_correct = True
        # only allow the user one margin of error, otherwise too easy
        elif lookahead == 0 and index+1 < len(detected) and detected[index+1][0] == chord:
            is_correct = True
            lookahead += 1
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
    filename = "../data/clips/testclip.wav" 
    freq = 44100  
    q = queue.Queue()

    # added a small sleep to allow for getting ready to strum
    time.sleep(2)
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

    clip, sr = librosa.load('../data/clips/testclip.wav', sr=None)
    return clip, sr

def analyse():
    """
        Helper function used during notebook research that loads the clip saved by record(),
        calculate the average chroma and detect the chord

        :return: The detected chord and similarity score from detect_chord
    """
    clip, samplerate = librosa.load('../data/clips/testclip.wav', sr=None)
    chroma = librosa.feature.chroma_cqt(y=clip, sr=samplerate)
    avg_chroma = np.mean(chroma, axis=1)
    return detect_chord(avg_chroma)