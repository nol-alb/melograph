# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 22:43:11 2021

@author: thiag
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os


fs = 44100
blockSize = 1024
paddedBlockSize = blockSize*4


binSize = (fs/2)/paddedBlockSize
freqBin = np.arange(binSize, (paddedBlockSize+1)*binSize, binSize)

wind=np.kaiser(5,2)

n = np.arange(8,77,1)
fn = 440*2**((n-49)/12)

noteMatrix = np.zeros((len(fn),len(freqBin)))
fnBin=np.zeros((5,len(fn)))
for ind in range(5):
    fnBin[ind] = np.round(fn*(ind+1)/binSize)-1

fnBin=fnBin.T
        
for note in range(noteMatrix.shape[0]):
    for harm in fnBin[note,:]:
         noteMatrix[note,int(harm-2):int(harm+3)]=wind
        
 
        
