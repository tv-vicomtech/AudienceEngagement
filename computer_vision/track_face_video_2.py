import cv2
import dlib
import face_recognition
import cvlib as cv

import threading
import time

trackingQuality_threshold = 7
n_frames_to_detect = 10
min_confidence = 0.55
up_sample=1
method="hog"

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
output_movie = cv2.VideoWriter('data/track_video/face_recognition_face.avi', fourcc, 29.97, (frame_width, frame_height))

def detectAndTrackMultipleFaces():

    frame_number = 0
    currentFaceID = 0

    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    rectangleColor = (0,0,255)
    faceTrackers = {}
    start= time.time()
    try:
        while True:
            rc, fullSizeBaseImage = input_movie.read()
            if not rc:
                print("Could not read frame")
                break

            baseImage = cv2.resize( fullSizeBaseImage, (int(frame_width), int(frame_height)))
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

                face_locations = face_recognition.face_locations(baseImage,up_sample,method)

                for (top, right, bottom, left) in face_locations:
                    x = left
                    y = top
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

                    if ((matchedFid is None)):

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
    print("tiempo{}".format(end))
    cv2.destroyAllWindows()
    exit(0)

if __name__ == '__main__':
    detectAndTrackMultipleFaces()
