"""Example program to demonstrate how to read a multi-channel time-series
from LSL in a chunk-by-chunk manner (which is more efficient)."""

import queue
from pylsl import StreamInlet, resolve_stream
import numpy as np
import matplotlib.pyplot as plt
from sympy import beta
from relaxation_score_calculator import power_spectrum, add_to_queue
from eeg_features import calc_band_features
from functools import partial
import time

MAX_SAMPLES_PER_CHUNK = 512

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

add_to_queue = partial(add_to_queue, max_size=MAX_SAMPLES_PER_CHUNK)
data_queue = np.empty((1, 4))
timestamps_queue = np.empty((1))




while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    chunk, timestamps = inlet.pull_chunk(max_samples=MAX_SAMPLES_PER_CHUNK)
    time.sleep(0.1)
    
    # add to queue \ fixed window buffer
    if len(timestamps):
        data_queue = add_to_queue(data_queue, np.array(chunk))
        timestamps_queue = add_to_queue(timestamps_queue, np.array(timestamps))

    # print(len(timestamps), timestamps_queue.shape[0])

    if len(timestamps_queue) == MAX_SAMPLES_PER_CHUNK:
        # Do stuff here
        freq, spectrum = power_spectrum(data_queue, timestamps_queue)

        # TODO add smoothing for spectrum
       
        # calc features
        band_features = calc_band_features(freq, spectrum)
        print(band_features)
