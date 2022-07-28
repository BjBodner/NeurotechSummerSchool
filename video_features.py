from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from eye_tracking_feature import EyeTrackingFeature
import pandas as pd
from motion_feature import MotionFeature

VIDEO_COLUMNS = ["motion_feature", "eye_tracking_feature"]

class VideoFeatures:
    def __init__(self):
        self.motion_feature = MotionFeature()
        self.eye_tracking_feature = EyeTrackingFeature()
        self.vs = VideoStream(src=0).start()
        self.print_freq = 10
        self.i = 0
        time.sleep(2.0)

    def __call__(self):
        # loop over the frames of the video
        # while True:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = self.vs.read()

        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            raise ValueError("frame not found")

        motion_feature_ = self.motion_feature(frame)
        eye_tracking_feature_ = self.eye_tracking_feature(frame)

        self.i += 1
        if self.i == self.print_freq:
            print(motion_feature_, eye_tracking_feature_)
            self.i = 0


        video_features = pd.DataFrame([motion_feature_, eye_tracking_feature_], index=VIDEO_COLUMNS).T
        return video_features