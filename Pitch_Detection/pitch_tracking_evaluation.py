import os
import glob
import math
import numpy as np
import scipy as sp
from scipy import signal
import pandas as pd

import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread

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
    """
    reference code from Audio Content Analysis course
    """
    # allocate memory
    numBlocks = int(np.ceil(x.size / hopSize))
    xb = np.zeros([numBlocks, blockSize])

    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return xb, t



# def thiago_get_f0_from_acf (r, fs):
    
#     peakind, _ = signal.find_peaks(r, distance=30)
#     highPeakind = np.argmax(r[peakind])
    
#     t0 = peakind[highPeakind]
#     print(t0)
#     f0 = -1
#     try:
#         if(t0 != 0):
#             f0 = fs/t0
#     except:
#         print(f"t0 value is: {t0}")
#         f0 = -1
#     return f0

def get_f0_from_acf (r, fs):
    eta_min = 1
    afDeltaCorr = np.diff(r)
    eta_tmp = np.argmax(afDeltaCorr > 0)
    eta_min = np.max([eta_min, eta_tmp])
    f = np.argmax(r[np.arange(eta_min + 1, r.size)])
    f = fs / (f + eta_min + 1)
    return (f)






def get_f0_from_nsdf (d, fs):
    
    peakind, _ = signal.find_peaks(d, distance=30)
    if (peakind.size!=0):
        highPeakind = np.argmax(d[peakind])
        t0 = peakind[highPeakind]
        f0 = fs/t0
    else:
        f0 = 0 
    
    return f0


def mod_acf2(x):
#     print(f"within mod_acf2: x={x}")
    try:
        if(np.max(x)):
            x=x/np.max(x)
    except:
        print("fata")
        x = 0
    x[x > 0] = 1
    x[x < 0] = -1
    x[x == 0] = 0
    return x

def extract_rms(xb):
    rmsDb = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        rmsDb[block] = np.sqrt(np.mean(np.square(xb[block])))
        threshold = 1e-5  # truncated at -100dB
        if rmsDb[block] < threshold:
            rmsDb[block] = threshold
        rmsDb[block] = 20 * np.log10(rmsDb[block])
    return rmsDb

def convert_freq2midi(fInHz, fA4InHz = 440):
        def convert_freq2midi_scalar(f, fA4InHz):
            if f <= 0:
                return 0
            else:
                return (69 + 12 * np.log2(f/fA4InHz))
            
        fInHz = np.asarray(fInHz)
        if fInHz.ndim == 0:
            return convert_freq2midi_scalar(fInHz,fA4InHz)
        midi = np.zeros(fInHz.shape)
        for k,f in enumerate(fInHz):
            midi[k] =  convert_freq2midi_scalar(f,fA4InHz)
#             print("midi[k]:",midi[k])    
        return (midi)
    
def eval_pitchtrack(estimateInHz, groundtruthInHz):
        if np.abs(groundtruthInHz).sum() <= 0:
            return 0
        # truncate longer vector
        if groundtruthInHz.size < estimateInHz.size:
            print("within if")
            estimateInHz = estimateInHz[np.arange(0,groundtruthInHz.size)]
        elif estimateInHz.size < groundtruthInHz.size:
            print("within else")
            groundtruthInHz = groundtruthInHz[np.arange(0,estimateInHz.size)]
        diffInCent = 100*(convert_freq2midi(estimateInHz) - convert_freq2midi(groundtruthInHz))
        print(f"est: {estimateInHz}, gt: {groundtruthInHz}, diffincent:{diffInCent}")
        rms = np.sqrt(np.mean(diffInCent**2))
        return (rms)
    
def eval_pitchtrack_v2(estimation,annotation): 
    errCentRms= eval_pitchtrack(estimation, annotation)
    return errCentRms

def comp_nccf(inputVector):
    
    inputLength = len(inputVector)
    n = np.ones(inputLength)
    paddedInput = np.pad(inputVector, (0,inputLength-1))
    for i in range(inputLength):        
        n[i] = np.sqrt(np.sum(inputVector**2)*np.sum((paddedInput[i:i+inputLength])**2))
        if n[i] == 0:
            n[i] = 1
    print(n)
    return n

def comp_FFTacf(inputVector):
    
    inputLength = len(inputVector)
    inFFT = np.fft.fft(inputVector, 2*inputLength)
    S = np.conj(inFFT)*inFFT
    c_fourier = np.real(np.fft.ifft(S))
    # c_fourier = (np.fft.ifft(S))
    r_fft = c_fourier[:(c_fourier.size//2)]
    
    return r_fft

def get_f0_from_nccf (r, n, fs):
    
    nccf = r/n
    peakind, _ = signal.find_peaks(nccf, distance=30)
    print("peakind:", peakind)
    if (peakind.size!=0):
        highPeakind = np.argmax(nccf[peakind])
        t0 = peakind[highPeakind]
        f0 = fs/t0
    else:
        f0 = 0
    return f0

def track_pitch_nccf(x,blockSize,hopSize,fs): 
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    f0 = np.zeros(len(timeInSec))    
    for idx, val in enumerate(xb):
        mod_val = mod_acf2(val) 
        n= comp_nccf(mod_val)
        r = comp_FFTacf(mod_val)
        print("Wthn track_pitch_nccf r:", r)
        f0[idx] = get_f0_from_nccf (r, n, fs)
    print(f"nccf: f0 {f0}")
    return [f0,timeInSec]

def get_d_from_r_m(r,m):
    return ((2*r)/m)

def comp_Msdf(inputVector):
    inputLength = len(inputVector)
    m = np.zeros(inputLength)
    paddedInput = np.pad(inputVector, (0,inputLength-1))
    for i in range(inputLength):        
        m[i] = np.sum((inputVector**2)+(paddedInput[i:i+inputLength])**2)
    return m

def track_pitch_nsdf(x,blockSize,hopSize,fs): 
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    f0 = np.zeros(len(timeInSec))    
    for idx, val in enumerate(xb):
        
        mod_val = mod_acf2(val) 
        m = comp_Msdf(mod_val)
        r = comp_acf(mod_val,False)
        d = get_d_from_r_m(r,m)
        f0[idx] = get_f0_from_nsdf (d, fs)
    print(f"nsdf: f0 {f0}")
    return [f0,timeInSec]
  
# def thiago_comp_acf(inputVector, bIsNormalized):
#     inputLength = len(inputVector)
#     r = np.zeros(inputLength)
#     paddedInput = np.pad(inputVector, (0,inputLength-1))
#     for i in range(inputLength):
#         r[i] = np.dot(inputVector, paddedInput[i:i+inputLength])
    
#     if bIsNormalized == True:
#         if (np.dot(inputVector,inputVector)!= 0):
#             r = r/np.dot(inputVector,inputVector)
# #         else:
# #             print("comp_acf r=", r)
            
#     return r

def comp_acf(inputVector, bIsNormalized = True):
    if bIsNormalized:
        norm = np.dot(inputVector, inputVector)
    else:
        norm = 1
    afCorr = np.correlate(inputVector, inputVector, "full") / norm
    afCorr = afCorr[np.arange(inputVector.size-1, afCorr.size)]
    return (afCorr)



def track_pitch_acf(x,blockSize,hopSize,fs):
    
    # get blocks
    [xb,t] = block_audio(x,blockSize,hopSize,fs)
    # init result
    f0 = np.zeros(xb.shape[0]
                  )
    # compute acf
    for n in range(0, xb.shape[0]):
        r = comp_acf(xb[n,:])
        
        f0[n] = get_f0_from_acf(r,fs)
    return (f0,t)

def track_pitch(x,blockSize,hopSize,fs,method):
    
    if method == 'acf':
        f0, timeInSec = track_pitch_acf(x,blockSize, hopSize, fs)
    elif method == 'nccf':
        f0, timeInSec = track_pitch_nccf(x,blockSize, hopSize, fs)
    elif method == 'nsdf':
        f0, timeInSec = track_pitch_nsdf(x,blockSize, hopSize, fs)
    """
    xb, t = block_audio(x, blockSize, hopSize, fs)
    rmsDb = extract_rms(xb)
    voicing_mask = create_voicing_mask(rmsDb, voicingThres)
    f0Adj = apply_voicing_mask(f0, voicing_mask)
    
    return f0Adj, timeInSec
    """
    return f0, timeInSec

def evaluate_trackpitch(pathToAudio, pathToGT):
    methods = ['acf','nccf','nsdf']
    iNumOfFiles = 0
    rms_avg = np.zeros((3,2))
    for met in range(len(methods)):
        rmsAvg = 0
        pfp=0
        pfn=0
        pfp_sum = 0
        pfn_sum = 0
        rmsAvg_sum = 0

        wav_file_names = [os.path.join(pathToAudio, _) for _ in os.listdir(pathToAudio) if _[-4:] == ".wav"]
        csv_file_names = [os.path.join(pathToGT, _) for _ in os.listdir(pathToGT) if _[-4:] == ".csv"]
        i = 0
        for wav,csv in zip(wav_file_names, csv_file_names):
            i+=1
            print(f'{i}, processing {wav} and {csv}')

            fs, x = ToolReadAudio(wav)
            f0Adj, timeInSec = track_pitch(x, 128, 128, fs, methods[met])
            
            df = pd.read_csv(csv)
            df.columns = ["time", "f0"]
            
            rmsAvg = eval_pitchtrack_v2(f0Adj, df['f0'])
            rmsAvg_sum += rmsAvg

        rms_avg[met] = rmsAvg_sum/len(csv_file_names)
    return rms_avg

#testing one by one

pathToAudio = "/Users/rhythmjain/Desktop/GTStuff/1-2/AudioContent/melograph/Bach10-mf0-synth/audio_stems/"

wav_file_names = [os.path.join(pathToAudio, _) for _ in os.listdir(pathToAudio) if _[-4:] == ".wav"]
fs, x = ToolReadAudio(wav_file_names[0])
track_pitch_acf(x, 128, 128, fs)
# track_pitch_nccf(x, 128, 128, fs)
# track_pitch_nsdf(x, 128, 128, fs)

#testing all together

path_audio = "/Users/rhythmjain/Desktop/GTStuff/1-2/AudioContent/melograph/Bach10-mf0-synth/audio_stems/"
path_anno = "/Users/rhythmjain/Desktop/GTStuff/1-2/AudioContent/melograph/Bach10-mf0-synth/annotation_stems"
# evaluate_trackpitch(path_audio, path_anno)

