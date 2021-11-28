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
    peaks = peak_picking_adaptive_threshold(x, median_len=16, offset_rel=0.05, sigma=4.0)
    peaks = np.int32(np.around(peaks*scale_factor*256*2))
    return peaks

def visualization(x,peaks):
    plt.figure()
    plt.plot(x)
    plt.plot(peaks,x[peaks],'x',c='r')
    plt.savefig('onsets.png')

