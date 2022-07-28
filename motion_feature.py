from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import math

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
# # if the video argument is None, then we are reading from webcam
# if args.get("video", None) is None:
# 	vs = VideoStream(src=0).start()
# 	time.sleep(2.0)
# # otherwise, we are reading from a video file
# else:
# 	vs = cv2.VideoCapture(args["video"])
# initialize the first frame in the video stream
firstFrame = None

i = 0
smoothed_motion_feature = 0
max_biffer_size = 10
print_freq = 10
smoothing = 0.90
frames_buffer = []
gaussian_blur_size = 7



# def sigmoid(x):
#   return 1 / (1 + math.exp(-x))

class MotionFeature:
    def __init__(self):
        self.frames_buffer = []
        self.scale = 3.
        self.baseline = 5.0
        self.smoothed_motion_feature = 0.0
        
    def transform(self, value):
        transformed_val = np.tanh(- (value - self.baseline) / self.scale)
        return transformed_val

    def __call__(self, frame):
        
        # resize the frame, convert it to grayscale, and blur it
        frame_ = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (gaussian_blur_size, gaussian_blur_size), 0)

        if len(self.frames_buffer) > max_biffer_size:
            self.frames_buffer.pop(0)
        self.frames_buffer.append(gray)

        firstFrame = self.frames_buffer[0]

        
        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)

        # calculate motion feature
        motion_feature = np.mean(frameDelta)
        self.smoothed_motion_feature = smoothing * self.smoothed_motion_feature + (1-smoothing) * motion_feature
        
        transformed_val = self.transform(self.smoothed_motion_feature)
        return transformed_val