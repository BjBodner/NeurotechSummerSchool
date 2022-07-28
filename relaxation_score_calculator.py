from eeg_features import EEGFeatures
from video_features import VideoFeatures
import pandas as pd
import numpy as np


class RelaxationScoreCalculator:
    def __init__(self):
        self.eeg_features = EEGFeatures()
        self.video_features = VideoFeatures()

    def calc_score(self):
        eeg_features_ = self.eeg_features()
        video_features_ = self.video_features()

        feature_vector = pd.concat([eeg_features_, video_features_], axis=1)
        relaxation_score = np.mean(feature_vector.values)
        return relaxation_score, feature_vector

