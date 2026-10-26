# **Final Year Project - Year 4 Software Development**
*By Mark Lambert - C00192497*

*Supervisor: Paul Barry*

---
# Guitar Chord Detection

Repo for Final year project for Year 4 Software Development: a tool for detecting guitar chords from audio, built with Python and librosa

# Installation
## Prerequisites
- Python 3.14.7 or newer
- Jupyter lab or your IDE of choice
- For live recording, a working microphone
- A tuned guitar!
- Install the required packages from the repo root using ```pip install -r requirements.txt```

## Clone the Repository
```bash
    git clone https://github.com/lambert-27/fretwave-chord-trainer.git
    cd fretwave-chord-trainer
```

## Project files

- `Iteration 1/` — First Iteration chord and onset detection code 
- `Documentation/` — project specification, first iteration report and diary

# Iteration 1

The first iteration explores the functionality of `librosa`, with the aim of using it in a practice tool that helps beginners work on the skill of changing cleanly between guitar chords

## Features Explored

- **Chord recognition:** `ChordDetect.py` analyses audio with librosa's chroma features and compares the average pitch-class profile with a small set of chord templates: C, G, D, Dm, E, Em, Am, F and A. It uses cosine similarity to choose the closest match
- **Audio input and recording:** The prototype can record from the computer's default microphone and save a WAV clip for analysis
- **Chord-change timing research:** The notebooks investigate onset detection as a way to measure when chords are played and how long changes take
- **Chord Prompting:** The final release goal, explores user interaction with the chord detection algorithms. Prints three random chords, begins the recording process and when finished, detects the chords and if the strummed chords matched the prompted chords



This iteration is a research prototype, intended to be explored using the notebooks. Its chord template set is small and focuses on beginner-friendly chords

## Running the prototype

The notebooks in this folder can be opened in JupyterLab or another Jupyter-compatible IDE.

Open `ChordPromptNotebook.ipynb` to run the chord prompt: it shows three chord diagrams, records your strumming, then reports whether each detected chord matched the prompted one

## Files

- `iteration_1_librosa_research.ipynb` — initial experimentation with librosa and prototype development
- `ChordDetect.py` — Chord detection prototype extracted from research notebook, includes functions defined for recording, chord and onset detection
- `ChordPromptNotebook.ipynb` - Prints three random chord images shown as chord diagrams to the user, begins recording and on completion analyses
- `OnsetDetectionResearch.ipynb` and `OnsetDetectionApplied.ipynb` — research and experiments with librosa's `onset_detect` function
- - `ChordDetectTest.py` — unit tests for `ChordDetect.py`, run with `python -m unittest ChordDetectTest` from the `Iteration 1` folder (WIP)
- `clips/` — audio clips used for testing and development, and the folder where recordings are saved
- `chords/` - 9 chord diagram jpg files used in `ChordPromptNotebook.ipynb` to prompt the user with three random chords 

## Author

**Mark Lambert C00192497**
- Email: [marklambert123@gmail.com](mailto:marklambert123@gmail.com)
- College Email: [c00192497@setu.ie](mailto:c00192497@setu.ie)
