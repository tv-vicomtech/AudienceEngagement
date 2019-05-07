import cv2
import dlib
import os
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image
import time
from keras import backend as K
from keras.models import load_model
import cvlib as cv
import threading
from cvlib.object_detection import draw_bbox
from ctypes import *
import glob
import math
import random
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_eval

trackingQuality_threshold = 8
n_frames_to_detect = 45
min_confidence = 0.75

input_movie = cv2.VideoCapture("data/test_videos/hamilton_clip.mp4")

if not input_movie.isOpened():
    print("Could not open video file")
    exit()

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(input_movie.get(3))
frame_height = int(input_movie.get(4))
OUTPUT_SIZE_WIDTH = frame_width
OUTPUT_SIZE_HEIGHT = frame_height
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('data/track_video/Yolo_keras_object.avi', fourcc, 29.97, (frame_width, frame_height))

width = np.array(frame_width, dtype=float)
height = np.array(frame_height, dtype=float)
image_shape = (height, width)
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
yolo_model = load_model("model_data/yolov2.h5")

yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
sess = K.get_session()

def detectAndTrackMultipleFaces():

    frame_number = 0
    currentFaceID = 0

    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    rectangleColor = (0,0,255)
    faceTrackers = {}
    model_image_size = (608, 608)
    start=time.time()
    try:
        while True:
            
            rc, fullSizeBaseImage = input_movie.read()
            if not rc:
                print("Could not read frame")
                break

            #baseImage = fullSizeBaseImage
            baseImage = cv2.resize( fullSizeBaseImage, ( 608, 608))
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

                #resized_image = baseImage.resize((608, 608), Image.BICUBIC)
                baseImage_data = np.array(baseImage, dtype='float32')
                baseImage_data /= 255.
                baseImage_data = np.expand_dims(baseImage_data, 0)  # Add batch dimension.



#baseImage, baseImage_data = preprocess_image(baseImage, model_image_size = (608, 608))
                out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:baseImage_data,K.learning_phase(): 0})

                for i,c in reversed(list(enumerate(out_classes))):
                    top, left, bottom, right = out_boxes[i]

                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(608, np.floor(bottom + 0.5).astype('int32'))
                    right = min(608, np.floor(right + 0.5).astype('int32'))

                    x = left
                    y = bottom
                    w = right-left
                    h = bottom-top
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

                    if ((matchedFid is None) and (out_scores[i] > min_confidence)):

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
