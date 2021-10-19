# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os


def block_audio(x,blockSize,hopSize,fs):
    
    inLen = len(x)
    nBlock = int(np.ceil((inLen-blockSize)/hopSize)+1)

    
    xb = np.zeros((nBlock,blockSize))
    timeInSample = np.arange(0, hopSize*nBlock, hopSize)
    timeInSec = timeInSample/fs
                  

    for i in range(len(timeInSec)):

        if i == len(timeInSec)-1:
            zeroPad = blockSize - len(x[int(timeInSample[i]):])
            xb[i] = np.pad(x[int(timeInSample[i]):], (0,zeroPad))
        else:
            xb[i] = x[int(timeInSample[i]):int(timeInSample[i]+blockSize)]        
  
    return [xb, timeInSec]
  
  
def comp_acf(inputVector, bIsNormalized):
    
    inputLength = len(inputVector)
    r = np.zeros(inputLength)
    paddedInput = np.pad(inputVector, (0,inputLength-1))
    for i in range(inputLength):
        r[i] = np.dot(inputVector, paddedInput[i:i+inputLength])
    
    if bIsNormalized == True:
        r = r/np.dot(inputVector,inputVector)
    
    return r
    
def comp_Msdf(inputVector):
    
    inputLength = len(inputVector)
    m = np.zeros(inputLength)
    paddedInput = np.pad(inputVector, (0,inputLength-1))
    for i in range(inputLength):        
        m[i] = np.sum((inputVector**2)+(paddedInput[i:i+inputLength])**2)
    
    return m

def comp_nccf(inputVector):
    
    inputLength = len(inputVector)
    n = np.zeros(inputLength)
    paddedInput = np.pad(inputVector, (0,inputLength-1))
    for i in range(inputLength):        
        n[i] = np.sqrt(np.sum(inputVector**2)*np.sum((paddedInput[i:i+inputLength])**2))
    
    return n

def comp_FFTacf(inputVector):
    
    inputLength = len(inputVector)
    inFFT = np.fft.fft(inputVector, 2*inputLength)
    S = np.conj(inFFT)*inFFT
    c_fourier = np.real(np.fft.ifft(S))
    # c_fourier = (np.fft.ifft(S))
    r_fft = c_fourier[:(c_fourier.size//2)]
    
    return r_fft

def comp_amdf(inputVector):
    
    inputLength = len(inputVector)
    mdf = np.zeros(inputLength)
    paddedInput = np.pad(inputVector, (0,inputLength-1))
    for i in range(inputLength):        
        mdf[i] = np.sum(np.abs((inputVector)-(paddedInput[i:i+inputLength])))
    
    return mdf

def comp_sdf(inputVector):
    
    inputLength = len(inputVector)
    sdf = np.zeros(inputLength)
    paddedInput = np.pad(inputVector, (0,inputLength-1))
    for i in range(inputLength):        
        sdf[i] = np.sum(((inputVector)-(paddedInput[i:i+inputLength]))**2)
    
    return sdf
    
def get_f0_from_acf (r, fs):
    
    peakind, _ = signal.find_peaks(r, distance=30)
    highPeakind = np.argmax(r[peakind])
    
    t0 = peakind[highPeakind]
    f0 = fs/t0
    
    return f0

def get_f0_from_nsdf (d, fs):
    
    peakind, _ = signal.find_peaks(d, distance=30)
    highPeakind = np.argmax(d[peakind])
    
    t0 = peakind[highPeakind]
    f0 = fs/t0
    
    return f0

def get_f0_from_nccf (r, n, fs):
    
    nccf = r/n
    peakind, _ = signal.find_peaks(nccf, distance=30)
    highPeakind = np.argmax(nccf[peakind])
    
    t0 = peakind[highPeakind]
    f0 = fs/t0
    
    return f0

def get_d_from_r_m(r,m):
    return ((2*r)/m)

def mod_acf(x):
    
    x=x/np.max(x)
    x[np.abs(x) < 0.3] = 0
    return x

def bcf(x):
    
    x=x/np.max(x)
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

def mod_acf2(x):
    
    x=x/np.max(x)
    x[x > 0] = 1
    x[x < 0] = -1
    x[x == 0] = 0
    return x

def hwr(x):
    
    x=x/np.max(x)
    x=(x+np.abs(x))/2
    
    return x

def track_pitch_nccf(x,blockSize,hopSize,fs): 
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    f0 = np.zeros(len(timeInSec))    
    for idx, val in enumerate(xb):
        
        mod_val = mod_acf2(val) 
        n= comp_nccf(mod_val)
        r = comp_FFTacf(mod_val)
        try:
            f0[idx] = get_f0_from_nccf (r, n, fs)
        except:
            print(idx)
    
    return [f0,timeInSec]

def track_pitch_nsdf(x,blockSize,hopSize,fs): 
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    f0 = np.zeros(len(timeInSec))    
    for idx, val in enumerate(xb):
        
        mod_val = mod_acf2(val) 
        m = comp_Msdf(mod_val)
        r = comp_acf(mod_val,False)
        d = get_d_from_r_m(r,m)
        try:
            f0[idx] = get_f0_from_nsdf (d, fs)
        except:
            print(idx)
    
    return [f0,timeInSec]

def track_pitch_acf(x,blockSize,hopSize,fs):     
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    
    f0 = np.zeros(len(timeInSec))    
    for idx, val in enumerate(xb):
        val = mod_acf2(val)
        try:
            f0[idx] = get_f0_from_acf(comp_acf(val,True),fs)
        except:
            print(idx)


    f0 = np.concatenate((f0[0]*np.ones(1),f0))
    return [f0,timeInSec]
    
    
def convert_freq2midi(freqInHz):
    pitchInMIDI = 69+12*np.log2(freqInHz/440)
    
    return pitchInMIDI 

def eval_pitchtrack(estimateInHz, groundtruthInHz):
    estimateMidi = convert_freq2midi(estimateInHz)
    groundtruthMidi = convert_freq2midi(groundtruthInHz)
    
    errCentRms = 100*np.sqrt(np.mean((estimateMidi-groundtruthMidi)**2))
    
    return errCentRms

def run_evaluation(complete_path_to_data_folder):
    
    blockSize = 1024
    hopSize = 512
    
    path = complete_path_to_data_folder
    txt_list = []
    wav_list = []
    
    dirs = os.listdir(path)
    for i in dirs:
        if i.endswith('.txt'):
            txt_list.append(i)
        elif i.endswith('.wav'):
            wav_list.append(i)    
    
    errCentRms = np.zeros(len(txt_list))
    for ind,val in enumerate(txt_list):

        reference = np.loadtxt(path+val)
        truef0 = reference[:,2]

        [fs, data] = wavfile.read(path+wav_list[ind])
        [f0_acf,timeInSec_acf] = track_pitch_acf(data,blockSize,hopSize,fs)
        [f0_nsdf,timeInSec_nsdf] = track_pitch_nsdf(data,blockSize,hopSize,fs)
        [f0_nccf,timeInSec_nccf] = track_pitch_nccf(data,blockSize,hopSize,fs)
        f0_acf[f0_acf > 500] = 0
        f0_nsdf[f0_nsdf > 500] = 0
        f0_nccf[f0_nccf > 500] = 0


    #     truef0_new = np.array([])
    #     f0_new = np.array([])

        
    #     for ind_f0,val_f0 in enumerate(truef0):
    #         if val_f0 != 0:
    #             truef0_new = np.append(truef0_new, val_f0)
    #             f0_new = np.append(f0_new, f0[ind_f0])
        
    #     errCentRms[ind] = eval_pitchtrack(f0_new, truef0_new)
        
    OverallErrCentRms = np.mean(errCentRms)

    

    return [errCentRms,OverallErrCentRms, f0_acf, f0_nsdf, truef0]

       
#####################################
 
# fs = 44100
# f1 = 441
# f2 = 882
# t0 = 0
# t1 = 1
# t2 = 2

# samples1 = t1*fs
# samples2 = (t2-t1)*fs
# dur1 = np.arange(t0,t1,1/fs)
# dur2 = np.arange(t1,t2,1/fs)
# y1 = np.sin(2 * np.pi * f1 * dur1)
# y2 = np.sin(2 * np.pi * f2 * dur2)

# y = np.concatenate((y1,y2), axis=None)
# dur = np.concatenate((dur1,dur2), axis=None)

# [f0,timeInSec] = track_pitch_acf(y,1024,512,fs) 
# truef0_1 = 441*np.ones(int(np.ceil(44100/512)))
# truef0_2 = 882*np.ones(int(np.ceil((88200/512)-np.ceil(44100/512))))
# truef0 = np.concatenate((truef0_1,truef0_2), axis=None)

# Error = f0-truef0

# plt.plot(f0)
# plt.xlabel("Block Index")
# plt.ylabel("Frequency (Hz)")
# plt.show()

# plt.plot(Error)
# plt.xlabel("Block Index")
# plt.ylabel("Error (Hz)")
# plt.show()


# path = 'C:/Users/thiag/Documents/Gatech PhD/Audio Content Analysis/trainData/'
path = './Assignment 1/trainData/'
# [errCentRms, OverallErrCentRms, f0acf, f0nsdf, truef0] = run_evaluation(path)


# plt.plot(f0)
# plt.xlabel("Block Index")
# plt.ylabel("Estimated Frequency (Hz)")
# plt.show()

blockSize = 1024
hopSize = 512
    
txt_list = []
wav_list = []
    
dirs = os.listdir(path)
for i in dirs:
    if i.endswith('.txt'):
       txt_list.append(i)
    elif i.endswith('.wav'):
        wav_list.append(i) 

ind = 0
val = txt_list[ind]
reference = np.loadtxt(path+val)
truef0 = reference[:,2]

[fs, data] = wavfile.read(path+wav_list[ind])
[f0_acf,timeInSec_acf] = track_pitch_acf(data,blockSize,hopSize,fs)
[f0_nsdf,timeInSec_nsdf] = track_pitch_nsdf(data,blockSize,hopSize,fs)
[f0_nccf,timeInSec_nccf] = track_pitch_nccf(data,blockSize,hopSize,fs)
