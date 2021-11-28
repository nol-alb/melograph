# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 21:02:30 2021

@author: thiag
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
import librosa
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import os
import time
import glob
import math

def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavread(cAudioFilePath)

    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        audio = x / float(2 ** (nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    return samplerate, audio

def block_audio(x, blockSize, hopSize, fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((np.zeros(hopSize), x, np.zeros(hopSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t

def block_chunk_audio(x,initPos,endPos,fs):
    # allocate memory
    xChunk = np.zeros(endPos-initPos)
    xChunk = x[initPos:endPos]
    return xChunk

def mod_acf2(x):
    
    x=x/np.max(x)
    x[x > 0] = 1
    x[x < 0] = -1
    x[x == 0] = 0
    return x

def comp_FFTacf(inputVector):
    
    inputLength = len(inputVector)
    inFFT = np.fft.fft(inputVector, 2*inputLength)
    S = np.conj(inFFT)*inFFT
    c_fourier = np.real(np.fft.ifft(S))
    # c_fourier = (np.fft.ifft(S))
    r_fft = c_fourier[:(c_fourier.size//2)]
    
    return r_fft

def comp_FFTnccf(inputVector):
    
    inputLength = len(inputVector)
    n = np.zeros(inputLength)
    sqInput = inputVector**2
          
    n = np.sqrt(comp_FFTacf(sqInput))
    
    return n

def get_f0_from_nccf(r, n, fs):
    
    padding = 5
    
    try:
        # nccf = r/n
        nccf = r/n
    except:
        # print(n=0)
        pass
    

    nccf_inter = interpolate(nccf, padding)

    
    peakind_nccf, _ = signal.find_peaks(nccf_inter, distance=45)
    highPeakind_nccf = np.argmax(nccf_inter[peakind_nccf])
    t0_nccf = peakind_nccf[highPeakind_nccf]

    t0 = t0_nccf
    f0 = padding*fs/t0
    
    return f0

def interpolate(y, padding):
    ySize = y.shape[0]
    x = np.linspace(0, ySize, num=ySize, endpoint=True)
    f = interp1d(x, y, kind='quadratic')
    
    xnew = np.linspace(0, ySize, num=ySize*padding, endpoint=True)
    y_interpolated = f(xnew)
    
    return y_interpolated

def extract_rms(x):
    return np.clip((10*np.log10(np.mean(x**2, axis=1))), -100, 0)

def extract_rms_chunk(x):
    return np.clip((10*np.log10(np.mean(x**2))), -100, 0)

def track_pitch_nccf(x,blockSize,hopSize,fs): 
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    f0 = np.zeros(len(timeInSec))    
    for idx, val in enumerate(xb):
        
        mod_val = mod_acf2(val) 
        n = comp_FFTnccf(mod_val)
        r = comp_FFTacf(mod_val)
        try:
            f0[idx] = get_f0_from_nccf(r, n, fs)
        except:
            # print(idx)
            pass
    
    _f0 = f0.copy()
    comparison = f0.copy()
    for i in reversed(range(1, f0.shape[0]-1)):
        if comparison[i] < 40:
            comparison[i] = comparison[i+1]
    # Smoothing and compare to eliminate some occasional octave errors
    smooth_over = 4
    comparison = np.convolve(comparison, np.ones(smooth_over * 2 + 1) / (smooth_over * 2 + 1), mode='valid')
    for i in range(smooth_over, _f0.shape[0] - smooth_over):
        if abs(_f0[i]/2-comparison[i - smooth_over]) < abs(_f0[i] - comparison[i - smooth_over]):
            _f0[i] = _f0[i]/2
        if abs(_f0[i]*2-comparison[i - smooth_over]) < abs(_f0[i] - comparison[i - smooth_over]):
            _f0[i] = _f0[i]*2
            
    
    return [f0,timeInSec]
    
        
def run_on_chunk(x,initPos,endPos, blockSize, hopSize, fs):
    f0, timeInSec = track_pitch_nccf(x[initPos:endPos], blockSize, hopSize, fs)
    return f0    

def onset_pitch_tracking(x,o, rmsThreshold):
    smallBlockSize = 1024
    smallHopSize = 512
    nChunks = o.shape[0]
    f0 = np.zeros(nChunks)
    for i in range(1, nChunks):
        initPos = o[i-1]
        endPos = o[i]
        rms = extract_rms_chunk(x[initPos:endPos])
        if rms > rmsThreshold:
            f0_temp = run_on_chunk(x,initPos,endPos,smallBlockSize,smallHopSize,fs)
            f0[i-1] = np.median(f0_temp)
    
    return f0

def freq2MIDI(f0,a0):
    f0_ = np.copy(f0)
    popIdx = np.zeros(0)
    for idx, val in enumerate(f0):
        if val == 0:
            popIdx = np.append(popIdx,idx)
    f0_ = np.delete(f0,popIdx.astype(int))
    pitchInMIDI = np.round(69 + 12 * np.log2(f0_/a0))
    return pitchInMIDI
    
    
path = 'C:/Users/thiag/Documents/Github/melograph/Audios/WholeToneWurl.wav'
[fs,x] = ToolReadAudio(path)
o = librosa.onset.onset_detect(x, sr=44100, hop_length=256, backtrack=True, units='samples')

f0 = onset_pitch_tracking(x,o, -20)
pitchInMIDI = freq2MIDI(f0,440)

