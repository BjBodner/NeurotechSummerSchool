# from src.eeg_features import EEGFeatures
# from video_features import VideoFeatures
# import pandas as pd
# import numpy as np

# class RelaxationScoreCalculator:
#     def __init__(self):
#         self.eeg_features = EEGFeatures()
#         self.video_features = VideoFeatures()

#     def calc_score(self):
#         eeg_features_ = self.eeg_features()
#         video_features_ = self.video_features()

#         feature_vector = pd.concat([eeg_features_, video_features_], axis=1)
#         relaxation_score = np.mean(feature_vector.values)
#         return relaxation_score, feature_vector


# class PrintLogger:
#     def __init__(self, print_freq=10):
#         self.i = 0

#     def log(self, relaxation_score, feature_vector):
#         self.i += 1
#         if self.i == 10:
#             print(f"relaxation_score = {relaxation_score} \nfeature_vector =\n {feature_vector}\n\n")
#             self.i = 0


from relaxation_score_calculator import RelaxationScoreCalculator
from print_logger import PrintLogger

logger = PrintLogger()
relaxation_score_calculator = RelaxationScoreCalculator()

while True:

    relaxation_score, feature_vector = relaxation_score_calculator.calc_score()
    logger.log(relaxation_score, feature_vector)

    # i += 1
    # if i == print_freq:
    #     print(f"relaxation_score = {relaxation_score}, feature_vector = {feature_vector}")
    #     i = 0