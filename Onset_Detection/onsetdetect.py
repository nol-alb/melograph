import numpy as np
import scipy 
from scipy import signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
from scipy.signal import find_peaks
import pydub
import json

def convert_stere_mono(path):
    from pydub import AudioSegment
    sound = AudioSegment.from_wav(path)
    sound = sound.set_channels(1)
    sound.export(path, format="wav")
    return 0
def json_helper(path):
    json_file = path
    with open(json_file) as f:
        data = json.load(f)
    time = []
    duration = []
    value = []
    for i in data["annotations"][7]['data']:
        time.append(i["time"])
        duration.append(i["duration"])
        value.append(i["value"])
    return time,duration,value


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

def take_small_chunks(x,fs):
    timestamps=[]
    initial = np.size(x)
    initial1 = int(initial/2)
    final = initial1+30000
    timestamps =[initial1/fs,final/fs]

    return x,timestamps

def stp(x, N, H):
    # x:  Input signal
    # N:  Frame length
    # H:  Hopsize
    x_pad = np.concatenate((x, np.zeros(N)))
    num_windows = np.ceil(1 + (len(x) - N) / H)
    win_pos = np.arange(num_windows) * H
    idx = np.array([np.arange(w, w+N) for w in win_pos], dtype='int32')
    P = (x_pad[idx] ** 2).sum(axis=1) / N
    return P
def avg_stp(P,j):
    J = j
    Ptm=np.zeros_like(P)
    for m in range(P.size):
        Ptm[m] = np.sum(P[m:m+J])/(J+1)
    return Ptm
def peak_picking(N,scale_factor,threshold=0.001):  
    novelty =N
    pos = np.append(novelty, novelty[-1]) > np.insert(novelty, 0, novelty[0])
    neg = np.logical_not(pos)
    peaks = np.where(np.logical_and(pos[:-1], neg[1:]))[0]
    
    values = novelty[peaks]
    values /= np.max(values)
    peaks = peaks[values >= threshold]
    values = values[values >= threshold]
    peaks_idx = np.int32(np.round(peaks*256*scale_factor))
    return peaks_idx

def hpss(x,fs):
    f, t, Zxx = signal.spectrogram(x, fs,'hann', nperseg=1024)
    win_harm =31 #Horizontal
    win_perc = 31 #Vertical
    harm = np.empty_like(Zxx)
    harm[:] = median_filter(Zxx, size=(1,win_harm ), mode='reflect')
    perc = np.empty_like(Zxx)
    perc[:] = median_filter(Zxx, size=(win_perc, 1), mode='reflect') 
    Xx = np.greater(harm,perc)
    Xy = np.greater(perc,harm)
    Mx = np.empty_like(harm)
    Mx = np.where(Xx==True,1,0)
    My = np.where(Xy==True,1,0)
    Zxx_harm = Mx*Zxx
    Zxx_perc = My*Zxx
    fs, xrec = signal.istft(Zxx_perc, fs)
    xrec = xrec/max(xrec)
    scale_factor= x.size/xrec.size
    P = stp(xrec,1024,256)
    Ptm = avg_stp(P,5)
    Pref = P-Ptm
    N = np.max((np.zeros_like(P),Pref),axis=0)
    N = N/max(N)
    peaks_idx = peak_picking(N,scale_factor,threshold=0.001)
    timestamps = peaks_idx/fs
    return peaks_idx,peaks_idx/256,timestamps


def evaluate_note_onsets(path,time_stamps_predi,timestmp_gnd,timestamp_shift):
    time_stamps_predi = time_stamps_predi + timestamp_shift
    
    
    return rmse


