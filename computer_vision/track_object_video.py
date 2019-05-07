import cv2
import dlib

import cvlib as cv

import threading
import time
import argparse
import time
from imageai.Detection import ObjectDetection
from ctypes import *
import glob
import math
import random
import os 

trackingQuality_threshold = 8
n_frames_to_detect = 45
min_confidence = 0.75

input_movie = cv2.VideoCapture("data/Dabadaba/Cam_1/3.mp4")

if not input_movie.isOpened():
    print("Could not open video file")
    exit()

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(input_movie.get(3))
frame_height = int(input_movie.get(4))
OUTPUT_SIZE_WIDTH = frame_width
OUTPUT_SIZE_HEIGHT = frame_height
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('data/track_video/cvlib_object.avi', fourcc, 29.97, (frame_width, frame_height))

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def detectAndTrackMultipleFaces():

    frame_number = 0
    currentFaceID = 0

    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    rectangleColor = (0,0,255)
    faceTrackers = {}
    start=time.time()
    try:
        while True:
            
            rc, fullSizeBaseImage = input_movie.read()
            if not rc:
                print("Could not read frame")
                break
            fullSizeBaseImage = increase_brightness(fullSizeBaseImage, value=30)
            #baseImage = fullSizeBaseImage
            baseImage = cv2.resize( fullSizeBaseImage, ( frame_width, frame_height))
            resultImage = baseImage.copy()
            frame_number += 1

            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                break

            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[ fid ].update( baseImage )

                #If the tracking quality is good enough, we must delete
                #this tracker
                if trackingQuality < trackingQuality_threshold:
                    fidsToDelete.append( fid )

            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop( fid , None )

            if (frame_number % n_frames_to_detect) == 0:

                bbox, label, conf = cv.detect_common_objects(baseImage)

                for i,label in enumerate(label):
                    x = bbox[i][0]
                    y = bbox[i][1]
                    w = bbox[i][2]-bbox[i][0]
                    h = bbox[i][3]-bbox[i][1]
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    matchedFid = None

                    for fid in faceTrackers.keys():
                        tracked_position =  faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
                             ( t_y <= y_bar   <= (t_y + t_h)) and 
                             ( x   <= t_x_bar <= (x   + w  )) and 
                             ( y   <= t_y_bar <= (y   + h  ))):
                            matchedFid = fid

                    if ((matchedFid is None) and (conf[i] > min_confidence)):

                        print("Creating new tracker " + str(currentFaceID))

                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,dlib.rectangle( x-10,y-20,x+w+10,y+h+20))

                        faceTrackers[ currentFaceID ] = tracker
                        currentFaceID += 1

            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                cv2.rectangle(resultImage, (t_x, t_y),(t_x + t_w , t_y + t_h), rectangleColor ,2)

            cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", resultImage)
            output_movie.write(resultImage)

    except KeyboardInterrupt as e:
        pass
    end=time.time()-start
    print("time: {}".format(end))
    cv2.destroyAllWindows()
    exit(0)

if __name__ == '__main__':
    detectAndTrackMultipleFaces()
