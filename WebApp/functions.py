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

def ToolBlockAudio(afAudioData, iBlockLength, iHopLength):

    iNumOfBlocks = np.floor((afAudioData.shape[0] - iBlockLength) / iHopLength + 1).astype(int)

    if iNumOfBlocks < 1:
        return np.array([])
    return np.vstack([np.array(afAudioData[i*iHopLength:i*iHopLength+iBlockLength]) for i in range(iNumOfBlocks)])

def block_audio(x,blockSize,hopSize,fs):
    
    inLen = len(x)
    nBlock = int(np.ceil((inLen-blockSize)/hopSize)+1)
    # nBlock = int(np.ceil(inLen/hopSize))
    xb = np.zeros((nBlock,blockSize))
    timeInSample = np.arange(0, hopSize*nBlock, hopSize)
    timeInSec = timeInSample/fs
    for i in range(nBlock):
        if i == len(timeInSec)-1:
            zeroPad = blockSize - len(x[int(timeInSample[i]):])
            xb[i] = np.pad(x[int(timeInSample[i]):], (0,zeroPad))
        else:
            xb[i] = x[int(timeInSample[i]):int(timeInSample[i]+blockSize)]        
    return xb
  

def mod_acf2(x):
    # x = x/np.max(np.abs(x))
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

def smooth(x, kernelSize):
    kernel = np.ones(kernelSize)/kernelSize
    return np.convolve(x, kernel, mode='same')

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

def get_f0_from_nccf(r, n, fs):
    nccf = r/n
    peakind_nccf, _ = signal.find_peaks(nccf, distance=50, prominence=20, threshold=1)
    if (peakind_nccf.shape[0] != 0):
        highPeakind_nccf = np.argmax(nccf[peakind_nccf])
        targetIndex = peakind_nccf[highPeakind_nccf]
        
        interpolBorder = 3
        interpolY = interpolate(nccf[targetIndex-interpolBorder:targetIndex+interpolBorder+1], 3)
        interpolX = interpolate(np.arange(targetIndex-interpolBorder,targetIndex+interpolBorder+1), 3)
        interpolPeak = np.argmax(interpolY)
    
        t0_nccf = interpolX[interpolPeak]
        t0 = t0_nccf
        f0 = fs/t0
    
        return f0
    else:
        return 0

def HPSS(x,fs):
    f, t, Zxx = signal.spectrogram(x, fs,window=('hann'), nperseg=1024)
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
    peaks = peak_picking_adaptive_threshold(nov, median_len=16, offset_rel=0.01, sigma=4.0)
    # peaks = np.int32(np.around(peaks*scale_factor*256*2))
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

def track_pitch_nccf(x,blockSize,hopSize,fs, thresholdDb): 
    
    xb = block_audio(x,blockSize,hopSize,fs)
    xb.T[xb.T == 0] = 0.00001
    f0 = np.zeros(xb.shape[0])    
    for idx, val in enumerate(xb):
        
        mod_val = mod_acf2(val) 
        n = comp_FFTnccf(mod_val)
        r = comp_FFTacf(mod_val)
        f0[idx] = get_f0_from_nccf(r, n, fs)

    f0[f0>2000] = 0
    
    smoothOver = 20
    # _f0 = f0.copy()
    # comparison = f0.copy()
    # for i in reversed(range(1, f0.shape[0]-1)):
    #     if comparison[i] < 40:
    #         comparison[i] = comparison[i+1]
    ## Smoothing and compare to eliminate some occasional octave errors
    # comparison = np.convolve(comparison, np.ones(smoothOver * 2 + 1) / (smoothOver * 2 + 1), mode='valid')
    # for i in range(smoothOver, _f0.shape[0] - smoothOver):
    #     if abs(_f0[i]/2-comparison[i - smoothOver]) < abs(_f0[i] - comparison[i - smoothOver]):
    #         _f0[i] = _f0[i]/2
    #     if abs(_f0[i]*2-comparison[i - smoothOver]) < abs(_f0[i] - comparison[i - smoothOver]):
    #         _f0[i] = _f0[i]*2
    
    smoothedF0 = smooth(f0, smoothOver)
    rmsDb = extract_rms(xb)
    mask = create_voicing_mask(rmsDb, thresholdDb)
    masked_f0 = apply_voicing_mask(smoothedF0, mask)
    
    return masked_f0

def extract_rms(xb):
    return np.clip((10*np.log10(np.mean(xb**2, axis=1))), -100, 0)

def create_voicing_mask(rmsDb, thresholdDb):
    mask = np.zeros(rmsDb.shape[0])
    mask[np.where(rmsDb > thresholdDb)] = 1
    # mask[np.where(flux < fluxThreshold)] = 1
    return mask
    

def apply_voicing_mask(f0, mask):
    return f0 * mask

def freq2MIDI(f0,a0):
    f0_ = np.copy(f0)
    popIdx = np.zeros(0)
    for idx, val in enumerate(f0):
        if val == 0:
            popIdx = np.append(popIdx,idx)
    f0_ = np.delete(f0,popIdx.astype(int))
    pitchInMIDI = np.round(69 + 12 * np.log2(f0_/a0))
    return pitchInMIDI


def onset_note_tracking(x, oInBlocks, fs, rmsThreshold):
    smallBlockSize = 512
    smallHopSize = 128
    a0 = 440
    nChunks = oInBlocks.shape[0]
    f0 = np.zeros(nChunks)
    for i in range(1, nChunks):
        initPos = oInBlocks[i-1]
        endPos = oInBlocks[i]
        rms = extract_rms_chunk(x[initPos:endPos])
        if rms > rmsThreshold:
            f0_temp = track_pitch_nccf(x[initPos:endPos], smallBlockSize, smallHopSize, fs)
            f0[i-1] = np.median(f0_temp)
    midiNoteArray = freq2MIDI(f0,a0)
    return midiNoteArray
