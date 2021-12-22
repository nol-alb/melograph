# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:59:42 2021

@author: thiag
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import math


def ToolReadAudio(cAudioFilePath):
    samplerate, x = wavfile.read(cAudioFilePath)

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

        audio = x / np.max(x)

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
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t

def mod_acf2(x):
    
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
    r_fft[r_fft == 0] = 0.00001
    return r_fft

def comp_FFTnccf(inputVector):
    
    inputLength = len(inputVector)
    n = np.zeros(inputLength)
    sqInput = inputVector**2
    sqAcf = comp_FFTacf(sqInput)
    
    n = np.sqrt(np.abs(sqAcf))
    n[n==0]=0.00001
    
    return n

def smooth(x, kernelSize):
    kernel = np.ones(kernelSize)/kernelSize
    return np.convolve(x, kernel, mode='same')

def get_f0_from_nccf(r, n, fs):
    nccf = r/n
    
    peakind_nccf, _ = signal.find_peaks(nccf, distance=50, prominence=20)
    if (peakind_nccf.shape[0] != 0):
        highPeakind_nccf = np.argmax(nccf[peakind_nccf])
        targetIndex = peakind_nccf[highPeakind_nccf]
        t0_nccf = targetIndex
        t0 = t0_nccf
        f0 = fs/t0
    
        return f0
    else:
        return 0

def interpolate(y, padding):
    ySize = y.shape[0]
    x = np.linspace(0, ySize, num=ySize, endpoint=True)
    f = interp1d(x, y, kind='quadratic')
    xnew = np.linspace(0, ySize, num=ySize*padding, endpoint=True)
    y_interpolated = f(xnew)
    return y_interpolated

def interpolateSize(y, newSize):
    ySize = y.shape[0]
    x = np.linspace(0, ySize, num=ySize, endpoint=True)
    f = interp1d(x, y, kind='quadratic')
    xnew = np.linspace(0, ySize, num=newSize, endpoint=True)
    y_interpolated = f(xnew)
    y_interpolated[y_interpolated < 5] = 0
    return y_interpolated
    
    
def track_pitch_nccf(x,blockSize,hopSize,fs, thresholdDb): 
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    xb.T[xb.T == 0] = 0.00001

    rmsDb = extract_rms(xb)
    f0 = np.zeros(len(timeInSec))    
    for idx, val in enumerate(xb):
        
        mod_val = mod_acf2(val) 
        n = comp_FFTnccf(mod_val)
        r = comp_FFTacf(mod_val)
        f0[idx] = get_f0_from_nccf(r, n, fs)

    f0[f0>2000] = 0

    smoothOver = 1
    smoothedF0 = smooth(f0, smoothOver)
    
    f0_ = np.copy(smoothedF0)
    popIdx = np.zeros(0)
    popIdx = np.where(rmsDb < thresholdDb)[0]
    f0_ = np.delete(f0_,popIdx)
    
    return [f0_,timeInSec]

def track_pitch_nccf_eval(x,blockSize,hopSize,fs, thresholdDb): 
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    xb.T[xb.T == 0] = 0.00001

    # rmsDb = extract_rms(xb)
    f0 = np.zeros(len(timeInSec))    
    for idx, val in enumerate(xb):
        
        mod_val = mod_acf2(val) 
        n = comp_FFTnccf(mod_val)
        r = comp_FFTacf(mod_val)
        f0[idx] = get_f0_from_nccf(r, n, fs)

    f0[f0>2000] = 0

    smoothOver = 25
    smoothedF0 = smooth(f0, smoothOver)
    shift = np.ones(5)*smoothedF0[0]
    shiftedSmoothedF0 = np.concatenate((shift,smoothedF0)) 

    f0_ = shiftedSmoothedF0
    
    return [f0_,timeInSec]


def extract_rms(xb):
    return np.clip((10*np.log10(np.mean(xb**2, axis=1))), -100, 0)

def apply_voicing_mask(f0, mask):
    return f0 * mask

def onset_note_tracking(x, oInBlocks, fs, rmsThreshold):
    smallBlockSize = 1024
    smallHopSize = 128
    a0 = 440
    nChunks = oInBlocks.shape[0]
    f0 = np.zeros(nChunks)
    for i in range(1, nChunks):
        initPos = oInBlocks[i-1]
        endPos = oInBlocks[i]
        xChunk = x[initPos:endPos]
        rms = extract_rms_chunk(x[initPos:endPos])
        if rms > rmsThreshold:
            [f0_temp,t] = track_pitch_nccf(xChunk, smallBlockSize, smallHopSize, fs, rmsThreshold)
            vals, counts = np.unique(f0_temp, return_counts=True)
            f0[i-1] = vals[np.argwhere(counts == np.max(counts))[0]][0]
    midiNoteArray = freq2MIDI(f0,a0)
    return midiNoteArray, f0

def extract_rms_chunk(x):
    return np.clip((10*np.log10(np.mean(x**2))), -100, 0)

def freq2MIDI(f0,a0):
    f0_ = np.copy(f0)
    popIdx = np.zeros(0)
    for idx, val in enumerate(f0):
        if val == 0:
            popIdx = np.append(popIdx,idx)
    f0_ = np.delete(f0,popIdx.astype(int))
    pitchInMIDI = np.round(69 + 12 * np.log2(f0_/a0))
    return pitchInMIDI

