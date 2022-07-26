import numpy as np
import pandas as pd

ALPHA_BAND = [8, 12]    
THETA_BAND = [4, 8]

BAND_COLUMNS = [
    "alpha_electrode1", "alpha_electrode2", "alpha_electrode3", "alpha_electrode4",
    "theta_electrode1", "theta_electrode2", "theta_electrode3", "theta_electrode4"
]



def calc_band_features(freq, spectrum):
    alpha_idx = np.where((freq > ALPHA_BAND[0]) & (freq < ALPHA_BAND[1]))[0]
    theta_idx = np.where((freq > THETA_BAND[0]) & (freq < THETA_BAND[1]))
    alpha_features = np.mean(spectrum[alpha_idx], 0)
    theta_features = np.mean(spectrum[theta_idx], 0)
    band_features = np.concatenate((alpha_features, theta_features))

    band_features = pd.DataFrame(band_features, index=BAND_COLUMNS).T
    return band_features