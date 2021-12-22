# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:59:42 2021

@author: thiag
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import Onsetdet as onset
import Pitch_Tracking as pitch
# import time

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

def run_evaluation_Bach(pathAudioFolder, pathGTFolder):
    blockSize = 1024 
    hopSize = 128
    thresholdDb = -40
    audioFileList = glob.glob(pathAudioFolder +'/'+'*.wav')
    GTFileList = glob.glob(pathGTFolder +'/'+'*.csv')
    estimateInHz = np.array([])
    groundtruthInHz = np.array([])
    
    for idx, audioFile in enumerate(audioFileList):
        # startTime = time.time()
        fs, x = pitch.ToolReadAudio(audioFile)
        csv = np.genfromtxt(GTFileList[idx], delimiter=",")
        gtF0 = csv[:,1]
        [f0,timeInSec] = pitch.track_pitch_nccf_eval(x, blockSize, hopSize, fs, thresholdDb)
        interpolatedF0 = pitch.interpolateSize(f0,gtF0.shape[0])
        
        errCentRms, pfp, pfn = eval_pitchtrack_v2(interpolatedF0, gtF0)
        displayname = GTFileList[idx].split('\\')[-1]
        print(f"RMS error for {displayname} is {errCentRms:.2f} cents. False positive is {pfp*100:2f}%. False negtative is {pfn*100:2f}%.")
        
        estimateInHz = np.concatenate((estimateInHz, interpolatedF0), 0)
        groundtruthInHz = np.concatenate((groundtruthInHz, gtF0), 0)
        # executionTime = (time.time() - startTime)
        # print('Execution time in seconds: ' + str(executionTime))
    overallErrCentRms, overallPfp, overallPfn = eval_pitchtrack_v2(estimateInHz, groundtruthInHz)
    print(overallErrCentRms)
    print(overallPfp)
    print(overallPfn)
    return overallErrCentRms, overallPfp, overallPfn
    # return interpolatedF0, gtF0

def run_evaluation_Flute(fullpath_txt,fullpath_aud):
    blockSizeOnset = 512 
    hopSizeOnset = 256
    thresholdDb = -20
    txt = []
    for file in os.listdir(fullpath_txt):
            if (file.startswith("scale") & file.endswith('.txt')):
                txt.append(fullpath_txt+file)
    aud = []
    for file in os.listdir(fullpath_aud):
            if (file.startswith("scale") & file.endswith('.wav')):
                aud.append(fullpath_aud+file)
    for i,k in zip(txt,aud):
        gnd = np.genfromtxt(i,skip_header=4,skip_footer=1)
        gnd_onset = np.genfromtxt(i,skip_header=4,skip_footer=1)
        if (gnd.size == 0):
            with open(i) as f:
                lines = np.array(f.readlines())
                lines = lines[4:-1]
                tmp = [data.split(' ') for data in lines]
                gnd_onset = [i[0] for i in tmp]
                gnd = [i[2][:-1] for i in tmp]
                gnd = np.asarray(gnd).astype('float32')
                gnd_onset = np.asarray(gnd_onset).astype('float32')
        else:        
            gnd = gnd[:,2]
            gnd_onset = gnd_onset[:,0]
        fs, x = pitch.ToolReadAudio(k)
        peakInSamples = onset.onset_detect(x,blockSizeOnset,hopSizeOnset,fs)
        midiNoteArray,f0 = pitch.onset_note_tracking(x, np.int32(peakInSamples), fs, thresholdDb)
        print(gnd)
        print(f0)
    return f0, gnd


def run_evaluation(path):
    filelist = glob.glob(path+'/'+'*.wav')
    estimateInHz = np.array([])
    groundtruthInHz = np.array([])
    for files in filelist:
        fs, x = pitch.ToolReadAudio(files)
        filename = files.split('.')[0]
        txt = glob.glob(filename+'*.txt')
        data = open(txt[0])
        txt_read = np.loadtxt(data)
        f0, timeInSec = pitch.track_pitch_nccf(x, 1024, 512, fs, -40)
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


def run_on_file(path, blockSize, hopSize):
    fs, x = pitch.ToolReadAudio(path)
    f0, timeInSec = pitch.track_pitch_nccf(x, blockSize, hopSize, fs, -30)
    plt.figure()
    plt.plot(f0)    
    return [f0,timeInSec]


# just for testing
if __name__ == '__main__':
    
    # # Run Evaluation for the Bach dataset # #
    pathGTBachFolder = 'D:/Doutorado/Gatech/MDB-stem-synth/annotation_stems'
    pathAudioBachFolder = 'D:/Doutorado/Gatech/MDB-stem-synth/audio_stems'
    [overallErrCentRms, overallPfp, overallPfn] = run_evaluation_Bach(pathAudioBachFolder,pathGTBachFolder)
   
    # # Run Evaluation for the Flute dataset # #
    # pathAudioFluteFolder = '../melograph/flute-audio-labelled-database-AMT/'
    # pathGTFluteFolder = '../melograph/flute-audio-labelled-database-AMT/'
    # run_evaluation_Flute(pathGTFluteFolder,pathAudioFluteFolder)
    
    # # Run whole process for single file # #
    # pathAudioSolo = 'C:/Users/thiag/Documents/Github/melograph/Bach10-mf0-synth/audio_stems/01_AchGottundHerr_clarinet.RESYN.wav'
    # blockSize = 1024
    # hopSize = 256
    # thresholdDb = -20
    # fs, x = pitch.ToolReadAudio(pathAudioSolo)
    # peakInSamples = onset.onset_detect(x,blockSize,hopSize,fs)
    # midiNoteArray,f0 = pitch.onset_note_tracking(x, np.int32(peakInSamples), fs, thresholdDb)


# EOF