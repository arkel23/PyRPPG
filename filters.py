import cv2
import math
from scipy.sparse import spdiags, eye
from scipy.signal import butter, lfilter
import numpy as np

def normalization(signal):
    mean = np.mean(signal, axis=0)
    std_dev = np.std(signal, axis=0)
    signal_proc = (signal - mean)/std_dev
    return signal_proc

def detrend(signal, lamb):
    # An Advanced Detrending Method With Application to HRV Analysis
    n_rows = signal.size
    # construct I
    I = eye(n_rows)
    # construct d2
    d2aux = np.ones((n_rows-2, 1), dtype=int)
    d2aux2 = np.array([1, -2, 1], dtype=int).reshape((-1, 1)).T
    diags = np.arange(0, 3, 1, dtype=int)
    D2 = spdiags(np.matmul(d2aux, d2aux2).T, diags, n_rows-2, n_rows).toarray()
    # detrended signal
    inter_prod = I - np.linalg.pinv(I + (lamb**2)*np.matmul(D2.T, D2))
    signal_proc = (I - inter_prod)@signal
    
    return signal_proc

def movingAverage(signal, filter_size):
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
    signal_proc = cv2.blur(signal, (filter_size, filter_size))
    return signal_proc

def butter_bandpass(lowcut, highcut, fs, order=5):
    # low and high frequencies as fraction of sampling frequency fs
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    # # https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    signal_proc = lfilter(b, a, signal)
    return signal_proc