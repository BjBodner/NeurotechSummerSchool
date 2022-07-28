import numpy as np
import pandas as pd
import queue
from pylsl import StreamInlet, resolve_stream
import numpy as np
import matplotlib.pyplot as plt
from sympy import beta
# from src.utils import power_spectrum, add_to_queue
from functools import partial
import time

MAX_SAMPLES_PER_CHUNK = 512


def power_spectrum(signal, timestamps):
    dt = np.mean(np.diff(timestamps))
    window_size = timestamps[-1] - timestamps[0]

    n = signal.shape[0]
    fft_signal = np.fft.fft(signal, axis=0)
    PSD = np.abs(fft_signal)
    window_size = signal.shape[0] * dt
    t = np.arange(0, window_size, dt)
    freq = (1/(dt*n)) * np.arange(n)
    L = np.arange(n // 2)
    return freq[L], PSD[L]

def add_to_queue(queue, new_data):
    queue = np.append(queue, new_data, 0)
    return queue[-MAX_SAMPLES_PER_CHUNK:]
# 
# add_to_queue = partial(add_to_queue, max_size=MAX_SAMPLES_PER_CHUNK)


ALPHA_BAND = [8, 12]    
THETA_BAND = [4, 8]

BAND_COLUMNS = [
    "alpha_electrode1", "alpha_electrode2", "alpha_electrode3", "alpha_electrode4",
    "theta_electrode1", "theta_electrode2", "theta_electrode3", "theta_electrode4"
]

MEAN_BAND_COLUMNS = [
    "mean_alpha_power", "mean_theta_power"
]

class EEGFeatures:
    def __init__(self):
        streams = resolve_stream('type', 'EEG')

        # create a new inlet to read from the stream
        self.inlet = StreamInlet(streams[0])

        self.data_queue = np.empty((1, 4))
        self.timestamps_queue = np.empty((1))

        self.alpha_baseline = 300
        self.alpha_scale = 500
        self.theta_baseline = 300
        self.theta_scale = 1000

    def transform(self, value, baseline, scale):
        transformed_val = np.tanh((value - baseline) / scale)
        return transformed_val

    def __call__(self):
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        chunk, timestamps = self.inlet.pull_chunk(max_samples=MAX_SAMPLES_PER_CHUNK)
        time.sleep(0.1)
        
        # add to queue \ fixed window buffer
        if len(timestamps):
            self.data_queue = add_to_queue(self.data_queue, np.array(chunk))
            self.timestamps_queue = add_to_queue(self.timestamps_queue, np.array(timestamps))


        if len(self.timestamps_queue) == MAX_SAMPLES_PER_CHUNK:
            # calc spectrum
            freq, spectrum = power_spectrum(self.data_queue, self.timestamps_queue)
        
            # calc features
            alpha_feature = self.calc_alpha_features(freq, spectrum)
            transformed_alpha_feature = self.transform(alpha_feature, self.alpha_baseline, self.alpha_baseline)
            theta_feature = self.calc_theta_features(freq, spectrum)
            transformed_theta_feature = self.transform(theta_feature, self.theta_baseline, self.theta_scale)

            # concatenate features
            band_features = pd.DataFrame([transformed_alpha_feature, transformed_theta_feature], index=MEAN_BAND_COLUMNS).T

        else:
            band_features = pd.DataFrame(np.zeros(2), index=MEAN_BAND_COLUMNS).T


        return band_features

    @staticmethod
    def calc_alpha_features(freq, spectrum):
        alpha_idx = np.where((freq > ALPHA_BAND[0]) & (freq < ALPHA_BAND[1]))[0]
        alpha_features = np.mean(np.mean(spectrum[alpha_idx], 0))
        return alpha_features

    @staticmethod
    def calc_theta_features(freq, spectrum):
        theta_idx = np.where((freq > THETA_BAND[0]) & (freq < THETA_BAND[1]))[0]
        theta_features = np.mean(np.mean(spectrum[theta_idx], 0))
        return theta_features


# def calc_band_features(freq, spectrum):
#     alpha_idx = np.where((freq > ALPHA_BAND[0]) & (freq < ALPHA_BAND[1]))[0]
#     theta_idx = np.where((freq > THETA_BAND[0]) & (freq < THETA_BAND[1]))
#     alpha_features = np.mean(np.mean(spectrum[alpha_idx], 0))
#     theta_features = np.mean(np.mean(spectrum[theta_idx], 0))
#     # band_features = np.concatenate((alpha_features, theta_features))

#     band_features = pd.DataFrame([alpha_features, theta_features], index=MEAN_BAND_COLUMNS).T
#     return band_features