import numpy as np
import os, sys, librosa
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write
from matplotlib import pyplot as plt
import IPython.display as ipd
import math
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from scipy.ndimage import filters

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
    x = np.concatenate((np.zeros(hopSize), x, np.zeros(hopSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb

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
    r_fft = c_fourier[:(c_fourier.size//2)]
    return r_fft

def comp_FFTnccf(inputVector):
    
    inputLength = len(inputVector)
    n = np.zeros(inputLength)
    sqInput = inputVector**2
    n = np.sqrt(comp_FFTacf(sqInput))
    return n

def get_f0_from_nccf(r, n, fs):
    try:
        nccf = r/n
    except:
        pass
    peakind_nccf, _ = signal.find_peaks(nccf, distance=45)
    highPeakind_nccf = np.argmax(nccf[peakind_nccf])
    t0_nccf = peakind_nccf[highPeakind_nccf]
    t0 = t0_nccf
    f0 = fs/t0
    return f0

def HPSS(x,fs):
    f, t, Zxx = signal.spectrogram(x, fs,'hann', nperseg=1024)
    win_harm =31 #Horizontal
    win_perc = 31 #Vertical
    harm = np.empty_like(Zxx)
    harm[:] = median_filter(Zxx, size=(1,win_harm ), mode='reflect')
    perc = np.empty_like(Zxx)
    perc[:] = median_filter(Zxx, size=(win_perc, 1), mode='reflect')
    X = np.greater(harm,perc)
    Xy = np.greater(perc,harm)
    Mx = np.empty_like(harm)
    Mx = np.where(X==True,1,0)
    My = np.where(Xy==True,1,0)
    Zxx_harm = Mx*Zxx
    Zxx_perc = My*Zxx
    y, xrec = signal.istft(Zxx_perc, fs)
    xrec=xrec/max(xrec)
    scale_factor= x.size/xrec.size
    return Zxx_perc,xrec,scale_factor

def Novelty_HPSS_Spectral(x,fs):
     freq,time,scale_factor = HPSS(x,fs)
     Y1 = np.log(1 + 100 * np.abs(freq)) #  Logarithmic Compression for smoothning
     Y_diff = np.diff(Y1, n=1)
     Y_diff[Y_diff < 0] = 0
     nov = np.sum(Y_diff, axis=0)
     nov = np.concatenate((nov, np.array([0])))
     nov = nov/max(nov) 
     return nov,scale_factor


"""
Adaptive thresholding peak detection 
"""
def peak_picking_adaptive_threshold(x, median_len=16, offset_rel=0.05, sigma=4.0):
    offset = x.mean() * offset_rel #Additional offset used for adaptive thresholding
    x = filters.gaussian_filter1d(x, sigma=sigma) #Variance for Gaussian kernel used for smoothing the novelty function
    threshold_local = filters.median_filter(x, size=median_len) + offset #median filtering for adaptive thresholding 
    peaks = []
    for i in range(1, x.shape[0] - 1):
        if (x[i- 1] < x[i] and x[i] > x[i + 1]):
            if (x[i] > threshold_local[i]):
                peaks.append(i)
    peaks = np.array(peaks)
    return peaks

def Onset_detection(path):
    fs,x = ToolReadAudio(path)
    nov,scale_factor = Novelty_HPSS_Spectral(x,fs)
    peaks = peak_picking_adaptive_threshold(nov, median_len=16, offset_rel=0.05, sigma=4.0)
    scale_factor = x.size/nov.size
    peaks = np.int32(np.around(peaks*scale_factor))
    return peaks

def visualization(x,peaks):
    plt.figure()
    plt.plot(x)
    plt.plot(peaks,x[peaks],'x',c='r')
    plt.savefig('onsets.png')

def extract_rms_chunk(x):
    return np.clip((10*np.log10(np.mean(x**2))), -100, 0)

def track_pitch_nccf(x,blockSize,hopSize,fs): 
    xb = block_audio(x,blockSize,hopSize,fs)
    f0 = np.zeros(xb.shape[0])    
    for idx, val in enumerate(xb):
        mod_val = mod_acf2(val) 
        n = comp_FFTnccf(mod_val)
        r = comp_FFTacf(mod_val)
        try:
            f0[idx] = get_f0_from_nccf(r, n, fs)
        except:
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

def onset_note_tracking(x, o, fs, rmsThreshold):
    smallBlockSize = 1024
    smallHopSize = 512
    a0 = 440
    nChunks = o.shape[0]
    f0 = np.zeros(nChunks)
    for i in range(1, nChunks):
        initPos = o[i-1]
        endPos = o[i]
        rms = extract_rms_chunk(x[initPos:endPos])
        if rms > rmsThreshold:
            f0_temp = track_pitch_nccf(x[initPos:endPos], smallBlockSize, smallHopSize, fs)
            f0[i-1] = np.median(f0_temp)
    midiNoteArray = freq2MIDI(f0,a0)
    return midiNoteArray
