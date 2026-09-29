import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# STEP 1 — Load a single chord clip (A major) and inspect it

# Load the audio file as a waveform array, with its sample rate
# sr=None uses the files own sample rate
aMajor, sampleRate = librosa.load('clips/Amaj.wav', sr=None)

# Plot the raw waveform
librosa.display.waveshow(aMajor, sr=sampleRate)
plt.title('A major - Waveform')
plt.show()

# Compute the chromagram: maps audio waves onto a chromagram of 12 notes
# (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
chroma = librosa.feature.chroma_cqt(y=aMajor, sr=sampleRate)

# Visualise the chromagram — A major chord = A,C#,E
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sampleRate)
plt.colorbar()
plt.title('A major - Chromagram')
plt.show()

# STEP 2 — Define chord templates and a matching function

# 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
chord_templates = {
    'C':  [1,0,0,0,1,0,0,1,0,0,0,0],
    'G':  [0,0,1,0,0,0,0,1,0,0,0,1],
    'D':  [0,0,1,0,0,0,1,0,0,1,0,0],
    'Em': [0,0,0,0,1,0,0,1,0,0,0,1],
    'Am': [1,0,0,0,1,0,0,0,0,1,0,0],
    'F':  [1,0,0,0,0,1,0,0,0,1,0,0],
    'A':  [0,1,0,0,1,0,0,0,0,1,0,0],
}

def detect_chord(chroma_frame):
    """
    Compare a single 12-length chroma vector against every chord template
    using cosine similarity, and return the best-matching chord name plus
    its similarity score (0 to 1, higher = more confident).

    dot product / (length of a * length of b)
    Parameters: a - chroma_frame
                b - template
                a.b - np.dot(chroma_frame, a)
                ||a|| - np.linalg.norm(chroma_frame) [length of the vector]
                ||b|| - np.linalg.norm(template)
                cos0 - score
    a . b = ||a|| ||b|| cos0
    cos0 = (a . b) / ||a|| ||b||
    """

    best_match, best_score = None, -1
    for name, template in chord_templates.items():
        # Cosine similarity: dot product of the two vectors, divided by
        # the product of their magnitudes
        score = np.dot(chroma_frame, template) / (
            np.linalg.norm(chroma_frame) * np.linalg.norm(template)
        )
        if score > best_score:
            best_score, best_match = score, name
    return best_match, best_score

# STEP 3 — Call functions to detect audio for A major

# axis=1 as we want to take the average across each row
avg_chroma = np.mean(chroma, axis=1)
print(f"The chord is: {detect_chord(avg_chroma)}")