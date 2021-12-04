# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:59:42 2021

@author: thiag
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import time
from scipy.signal import butter, lfilter, freqz
import glob
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

        audio = x / float(2**(nbits - 1))

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

def extract_spectral_flux(xb):
    
    [nBlocks_,blockSize_] = xb.shape
    SpecFluxVector = np.zeros(nBlocks_)

    Hann = np.hanning(blockSize_)
    Spectrogram = np.abs(np.fft.fft(xb*Hann,2*blockSize_))
    Spectrogram_ = Spectrogram[:,:blockSize_]
    Spectrogram_1 = np.concatenate((np.zeros((1,blockSize_)),Spectrogram_),axis=0)
    
    SpecDiff = np.diff(Spectrogram_1, axis=0)
    SpecFluxVector = np.sqrt(np.sum(SpecDiff**2, axis=1))/(blockSize_/2)
            
    return SpecFluxVector 


def extract_rms(xb):
    return np.clip((10*np.log10(np.mean(xb**2, axis=1))), -100, 0)

def create_voicing_mask(rmsDb, thresholdDb):
    mask = np.zeros(rmsDb.shape[0])
    mask[np.where(rmsDb > thresholdDb)] = 1
    # mask[np.where(flux < fluxThreshold)] = 1
    return mask
    

def apply_voicing_mask(f0, mask):
    return f0 * mask


def comp_FFTacf(inputVector):
    
    inputLength = len(inputVector)
    inFFT = np.fft.fft(inputVector, 2*inputLength)
    S = np.conj(inFFT)*inFFT
    c_fourier = np.real(np.fft.ifft(S))
    # c_fourier = (np.fft.ifft(S))
    r_fft = c_fourier[:(c_fourier.size//2)]
    
    return r_fft

def calc_diff_func(x, r):
    return r[0] + np.flip(np.cumsum(np.flip(x**2))) - 2*r


def cmndf(d):
    return np.concatenate((np.zeros(1), d[1:]*np.arange(1, d.shape[0])/np.cumsum(d[1:])))


# YIN algorithm
def track_pitch_cmndf(x, blockSize, hopSize, fs, thresholdDb):
    thres = 0.4
    x = x/np.max(np.abs(x))
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    f0 = np.zeros(timeInSec.shape[0])
    for i in range(timeInSec.shape[0]):
        r = comp_FFTacf(xb[i, :])
        d = calc_diff_func(xb[i, :], r)
        cd = cmndf(d)
        minima = np.where((cd[1:-1] < cd[0:-2]) * (cd[1:-1] < cd[2:]))[0] + 1
        idx = minima[np.where(cd[minima] < thres)[0]]
        if idx.size == 0:
            f0[i] = 0
        else:
            target_idx = idx[np.argmin(cd[idx])]
            t0 = np.array([target_idx-1, target_idx, target_idx+1])/fs
            y0 = cd[[target_idx-1, target_idx, target_idx+1]]
            f = interp1d(t0, y0, kind='quadratic')
            tnew = np.linspace((target_idx-1)/fs, (target_idx+1)/fs, 17)
            ynew = f(tnew)
            f0[i] = 1/tnew[np.argmin(ynew)]
    _f0 = f0.copy()
    comparison = f0.copy()
    for i in reversed(range(1, f0.shape[0]-1)):
        if comparison[i] < 40:
            comparison[i] = comparison[i+1]
    # Smoothing and compare to eliminate some occasional octave errors
    smooth_over = 10
    comparison = np.convolve(comparison, np.ones(smooth_over * 2 + 1) / (smooth_over * 2 + 1), mode='valid')
    for i in range(smooth_over, _f0.shape[0] - smooth_over):
        if abs(_f0[i]/2-comparison[i - smooth_over]) < abs(_f0[i] - comparison[i - smooth_over]):
            _f0[i] = _f0[i]/2
        if abs(_f0[i]*2-comparison[i - smooth_over]) < abs(_f0[i] - comparison[i - smooth_over]):
            _f0[i] = _f0[i]*2
            
            
    rmsDb = extract_rms(xb)
    # specFlux = extract_spectral_flux(xb)
    mask = create_voicing_mask(rmsDb, thresholdDb)
    masked_f0 = apply_voicing_mask(_f0, mask)        
    return masked_f0, timeInSec


# def postProcessF0(f0_old):
#     f0Size = f0_old.shape[0]
#     newF0 = f0_old
    
#     for n in range(1,f0Size-1):
#         tolerance = 0.9
        
#         if(f0_old[n] >= (1+tolerance)*f0_old[n-1] and f0_old[n] <= (3-tolerance)*f0_old[n+1]):
#             newF0[n] = f0_old[n]/2
#             print("up")
#             print(n)
        
#         if(f0_old[n] <= f0_old[n-1]/(1+tolerance) and f0_old[n] >= f0_old[n+1]/(3-tolerance)):
#             newF0[n] = f0_old[n]*2
#             print(n)
#             print("down")
        
#     return newF0

def convert_freq2midi(freqInHz):
    a0 = 440
    pitchInMIDI = 69 + 12 * np.log2(freqInHz/a0)
    return pitchInMIDI

def eval_voiced_fp(estimation, annotation):
    return np.count_nonzero(estimation[np.where(annotation == 0)[0]])/np.where(annotation == 0)[0].size


def eval_voiced_fn(estimation, annotation):
    return np.where(estimation[np.nonzero(annotation)] == 0)[0].size/np.count_nonzero(annotation)

def eval_pitchtrack_v2(estimation, annotation):
    estimateInCents = 100 * convert_freq2midi(estimation[np.intersect1d(np.nonzero(estimation), np.nonzero(annotation))])
    groundtruthInCents = 100 * convert_freq2midi(annotation[np.intersect1d(np.nonzero(estimation), np.nonzero(annotation))])
    errCentRms = np.sqrt(np.mean((estimateInCents-groundtruthInCents)**2))
    return errCentRms, eval_voiced_fp(estimation, annotation), eval_voiced_fn(estimation, annotation)


def run_evaluation(path):
    filelist = glob.glob(path+'/'+'*.wav')
    estimateInHz = np.array([])
    groundtruthInHz = np.array([])
    for files in filelist:
        fs, x = ToolReadAudio(files)
        filename = files.split('.')[0]
        txt = glob.glob(filename+'*.txt')
        data = open(txt[0])
        txt_read = np.loadtxt(data)
        f0, timeInSec = track_pitch_cmndf(x, 1024, 512, fs, -40)
        errCentRms, pfp, pfn = eval_pitchtrack_v2(f0, txt_read[:, 2].T)
        displayname = files.split('\\')[-1]
        print(f"RMS error for {displayname} is {errCentRms:.2f} cents. False positive is {pfp*100:2f}%. False negtative is {pfn*100:2f}%.")
        plt.figure()
        plt.plot(timeInSec, f0,'.')
        plt.plot(txt_read[:, 0], txt_read[:, 2])
        plt.xlabel('Time/s')
        plt.ylabel('Frequency/Hz')
        plt.title(f'Detected Frequency for {displayname}')
        plt.legend(['Detected', 'Ground Truth'])
        plt.show()
        estimateInHz = np.concatenate((estimateInHz, f0), 0)
        groundtruthInHz = np.concatenate((groundtruthInHz, txt_read[:, 2].T), 0)
    return eval_pitchtrack_v2(estimateInHz, groundtruthInHz)

def run_on_file(path):
    fs, x = ToolReadAudio(path)
    f0, timeInSec = track_pitch_cmndf(x, 1024, 512, fs, -20)
    plt.plot(f0)
    return [f0,timeInSec]

# just for testing
# blockSize = 1024
# hopSize = 512

# fs = 44100
# f1 = 60
# f2 = 2000
# t0 = 0
# t1 = 1
# t2 = 2

# samples1 = t1*fs
# samples2 = (t2-t1)*fs
# dur1 = np.arange(t0,t1,1/fs)
# dur2 = np.arange(t1,t2,1/fs)

# y1 = signal.sawtooth(2 * np.pi * f1 * dur1)
# y2 = signal.sawtooth(2 * np.pi * f2 * dur2)
# y3 = ((np.random.rand(10000)/100000)-0.1)
   
# y = np.concatenate((y1,y2,y3), axis=None)

# data = y
# data = data/np.max(data)

# [xb, timeInSec] = block_audio(data,blockSize,hopSize,fs)

# start = time.time()
# [f0_nccf,timeInSec_nccf] = track_pitch_nccf(data,blockSize,hopSize,fs)
# end = time.time()
# print("The time of execution of NCCF is :", end-start)
# plt.plot(f0_nccf,".")

# f=f0_nccf

# just for testing
if __name__ == '__main__':
    path = "C:/Users/thiag/Documents/Github/melograph/Audios/WholeToneStrings.wav"
    blockSize = 1024
    hopSize = 512
    thresholdDb = -40
    # run_evaluation('C:/Users/thiag/Documents/Github/melograph/trainData')
    [f0,timeInSec] = run_on_file("C:/Users/thiag/Documents/Github/melograph/Audios/WholeToneStrings.wav")
    # plt.plot(f0)
    
    fs, x = ToolReadAudio(path)
    
# EOF