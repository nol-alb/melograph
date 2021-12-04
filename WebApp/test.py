# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:55:54 2021

@author: thiag
"""

import functions as func
import librosa
import numpy as np
import matplotlib.pyplot as plt

pathGT = 'C:/Users/thiag/Documents/Github/melograph/Bach10-mf0-synth/annotation_stems/01_AchGottundHerr_bassoon.RESYN.csv'
pathAudio = 'C:/Users/thiag/Documents/Github/melograph/Bach10-mf0-synth/audio_stems/01_AchGottundHerr_bassoon.RESYN.wav'
blockSize = 1024
hopSize = 128

# [fs,x] = func.ToolReadAudio(pathAudio)
# # o = librosa.onset.onset_detect(x, sr=fs, hop_length=hopSize, backtrack=True, units='samples')

# # midiNotes = func.onset_note_tracking(x, o, fs, -20)
# [_testF0,testF0] = func.track_pitch_nccf(x,blockSize,hopSize,fs)

# csv = np.genfromtxt(pathGT, delimiter=",")
# timeStanp = csv[:,0]
# gtF0 = csv[:,1]
# gtNotes = func.freq2MIDI(gtF0,440)

# # plt.plot(gtF0)
# plt.plot(testF0)
# plt.plot(_testF0)

onsets = func.Onset_detection(pathAudio)