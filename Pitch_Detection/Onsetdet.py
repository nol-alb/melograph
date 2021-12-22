import numpy as np
from scipy import signal
from scipy.io.wavfile import read as wavread
import math
from scipy.ndimage import median_filter
from scipy.ndimage import filters
from scipy.fftpack import fft

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
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t
def compute_hann_window(window_size):
    return 0.5*(1-(np.cos(2*np.pi*(np.arange(window_size)/window_size))))
def fourier(x):
    # Get Symmetric fft
    w = signal.windows.hann(np.size(x))
    windowed = x * w
    w1 = int((x.size + 1) // 2)
    w2 = int(x.size / 2)
    fftans = np.zeros(x.size)

    # Centre to make even function
    fftans[0:w1] = windowed[w2:]
    fftans[w2:] = windowed[0:w1]
    X = fft(fftans)
    magX = abs(X[0:int(x.size // 2 + 1)])
    return magX
def extract_spectral_flux(xb):
    magX = np.zeros((xb.shape[0], int(xb.shape[1]/2 + 1)))
    specflux = np.zeros((xb.shape[0]))
    magX[0] = fourier(xb[0])
    for block in np.arange(1, xb.shape[0]):
        magX[block] = fourier(xb[block])
        den = magX[block].shape[0]
        specflux[block] = np.sqrt(np.sum(np.square(magX[block] - magX[block-1])))/den
    return specflux

def compute_localaverage(nov,local_range):
        averages = np.zeros_like(nov)
        for j in range(nov.size):
            averages[j] = np.sum(nov[j:j+local_range])/(local_range+1)
        nov_1 =  nov - averages
        nov_1[nov_1<0]=0
        nov_1 = nov_1 / max(nov_1)
        return nov_1

def onset_detect(x,blocksize,hopsize,fs):
    xb,timeInSec = block_audio(x, blocksize, hopsize, fs)
    specflux = extract_spectral_flux(xb)
    #plt.figure()
    #plt.plot(specflux)
    avg = compute_localaverage(specflux,int(specflux.size/10))
    #plt.figure()
    #plt.plot(avg)
    x_main = filters.gaussian_filter1d(avg, 6)
    #plt.figure()
    #plt.plot(x_main)
    avg = compute_localaverage(x_main,int(specflux.size/10))
    #plt.figure()
    #plt.plot(avg)
    x_main= avg
    offset = np.mean(avg) * 0.05
    local = median_filter(x_main, size = 40) + offset
    #plt.figure()
    #plt.plot(x_main)
    #plt.plot(local)
    x_pushed = np.zeros(specflux.size+1)
    x_pushed[1:]=x_main
    peaks = []
    for i in range(1, x_pushed.shape[0] - 1):
        if (x_pushed[i- 1] < x_pushed[i] and x_pushed[i] > x_pushed[i + 1]):
            if (x_pushed[i] > local[i]):
                peaks.append(i-1)
    peaks = np.asarray(peaks)
    peaks = np.int32(np.around(peaks*(x.size/specflux.size)))
    peaks_diff = np.diff(peaks)
    peaks_time = peaks_diff/fs
    idx_del = np.where(peaks_time<0.05)[0]
    new_peaks = np.delete(peaks,idx_del)
    return new_peaks