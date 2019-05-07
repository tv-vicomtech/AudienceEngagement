import cv2
import dlib

import cvlib as cv

import threading
import time
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import argparse
from PIL import Image, ImageDraw
import time
from imageai.Detection import ObjectDetection
from ctypes import *
from PIL import Image, ImageDraw
import glob
import math
import cvlib as cv
import cv2
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
output_movie = cv2.VideoWriter('data/track_video/imageai_object.avi', fourcc, 29.97, (frame_width, frame_height))

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "model_data/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed="fastest")
custom_objects = detector.CustomObjects(person=True)

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

                detected_image_array, detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,output_type="array", input_type="array",input_image=baseImage, minimum_percentage_probability=(min_confidence*100))

                detection_2 = detections

                for i,detection_2 in enumerate(detection_2):
                    for key,value in detections[i].items():
                        print(key,value)
                        if(key=='name'): name=value
                        if(key=='percentage_probability'): probability=value
                        if(key=='box_points'): points=value
                        
                    x = points[0]
                    y = points[1]
                    w = points[2]-points[0]
                    h = points[3]-points[1]
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

                    if ((matchedFid is None) and (probability > min_confidence)):

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
    print("tiempo: {}".format(end))
    cv2.destroyAllWindows()
    exit(0)

if __name__ == '__main__':
    detectAndTrackMultipleFaces()
