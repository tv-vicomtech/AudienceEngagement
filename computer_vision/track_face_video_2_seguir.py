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
sec=5

input_movie = cv2.VideoCapture("data/test_videos/dinner.mp4")


if not input_movie.isOpened():
    print("Could not open video file")
    exit()

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(input_movie.get(3))
frame_height = int(input_movie.get(4))
OUTPUT_SIZE_WIDTH = frame_width
OUTPUT_SIZE_HEIGHT = frame_height
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('data/track_video/face_recognition_face_seguir.avi', fourcc, 29.97, (frame_width, frame_height))
listofcenters = []
centers = []
max_displacement = []
max_movement_single=[]

def detectAndTrackMultipleFaces():

    frame_number = 0
    currentFaceID = 0
    cont=0

    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    rectangleColor = (0,0,255)
    faceTrackers = {}
    start= time.time()
    try:
        while True:
            if int(time.time()-start)> ((cont+1)*5):
                cont+=1
                print("Total movement for the past {} seconds:".format(cont*5))
                for item in max_displacement:
                    print(item)
                print("Maximun movement for the past {} seconds:".format(cont*5))
                for item_2 in max_movement_single:
                    print(item_2)

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
                            centers=listofcenters[fid]
                            centers=[(x_bar,y_bar)]+centers
                            listofcenters[fid]=centers
                            max_distance=max_displacement[fid]
                            for (x,y) in centers:
                                distance=abs((pow(x_bar,2)+pow(y_bar,2))-(pow(t_x_bar,2)+pow(t_y_bar,2)))
                                max_displacement[fid]+=distance
                                if distance > max_movement_single[fid]:
                                    max_movement_single[fid]=distance

                    if ((matchedFid is None)):

                        print("Creating new tracker " + str(currentFaceID))

                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,dlib.rectangle( x-10,y-20,x+w+10,y+h+20))

                        faceTrackers[ currentFaceID ] = tracker
                        currentFaceID += 1
                        centers=[]
                        centers=[(x_bar,y_bar)]+centers
                        listofcenters.append(centers)
                        max_displacement.append(0)
                        max_movement_single.append(0)


            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h
                centers=listofcenters[fid]
                centers=[(t_x_bar,t_y_bar)]+centers
                listofcenters[fid]=centers
                max_distance=max_displacement[fid]
                for (x,y) in centers:
                    distance=abs((pow(x_bar,2)+pow(y_bar,2))-(pow(t_x_bar,2)+pow(t_y_bar,2)))
                    max_displacement[fid]+=distance
                    if distance > max_movement_single[fid]:
                        max_movement_single[fid]=distance

                cv2.rectangle(resultImage, (t_x, t_y),(t_x + t_w , t_y + t_h), rectangleColor ,2)

            for fid in faceTrackers.keys():
                values=listofcenters[fid]
                stop=0
                while stop < (len(values)-2):
                    x_1=int(values[stop][0])
                    y_1=int(values[stop][1])
                    x_2=int(values[stop+1][0])
                    y_2=int(values[stop+1][1])
                    # Plot the number in the list and set the line thickness.
                    if stop<10:
                        cv2.line(resultImage,(x_1,y_1),(x_2,y_2),(255,0,0),4)
                    elif stop < 40:
                        cv2.line(resultImage,(x_1,y_1),(x_2,y_2),(0,255,0),2)
                    elif stop < 100:
                        cv2.line(resultImage,(x_1,y_1),(x_2,y_2),(0,0,255),1)
                    stop+=1

            cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", resultImage)
            output_movie.write(resultImage)

    except KeyboardInterrupt as e:
        pass
    end=time.time()-start
    print("tiempo{}".format(end))
    print("Total movement for the past {} seconds:".format(end))
    for item in max_displacement:
        print(item)
    print("Maximun movement for the past {} seconds:".format(end))
    for item_2 in max_movement_single:
        print(item_2)

    cv2.destroyAllWindows()
    exit(0)

if __name__ == '__main__':
    detectAndTrackMultipleFaces()
