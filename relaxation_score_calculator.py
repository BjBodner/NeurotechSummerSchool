from eeg_features import EEGFeatures
from video_features import VideoFeatures
import pandas as pd
import numpy as np


class RelaxationScoreCalculator:
    def __init__(self, use_eeg=False):
        self.use_eeg = use_eeg
        self.eeg_features = EEGFeatures() if self.use_eeg else None
        self.video_features = VideoFeatures()

    def calc_score(self):
        if self.use_eeg:
            eeg_features_ = self.eeg_features()
        video_features_ = self.video_features()

        feature_vector = pd.concat([eeg_features_, video_features_], axis=1) if self.use_eeg else video_features_
        relaxation_score = np.mean(feature_vector.values)
        return relaxation_score, feature_vector

