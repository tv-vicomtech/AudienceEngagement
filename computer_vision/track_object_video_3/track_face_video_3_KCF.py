from __future__ import print_function
import cv2
import cvlib as cv
import time
import sys
from random import randint

input_movie = cv2.VideoCapture("../data/test_videos/hamilton_clip.mp4")

if not input_movie.isOpened():
    print("Could not open video file")
    exit()

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(input_movie.get(3))
frame_height = int(input_movie.get(4))
n_frames_to_detect = 10
frame_number = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('../data/track_video/KCF.avi', fourcc, 29.97, (frame_width, frame_height))

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker

trackerType = "KCF"
bboxes = []

# Grab a single frame of video
start=time.time()
ret, frame = input_movie.read()

face, confidence = cv.detect_face(frame)

for idx, f in enumerate(face):
    bbox=(f[0],f[1],(f[2]-f[0]),(f[3]-f[1]))
    bboxes.append(bbox)
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]

    # draw rectangle over face
    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

multiTracker = cv2.MultiTracker_create()
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)
output_movie.write(frame)
while input_movie.isOpened():
    frame_number += 1
    ret, frame = input_movie.read()
    if not ret:
        print("Could not read frame")
        break
    if frame_number==length:
        break
    if(frame_number % n_frames_to_detect)==0:
        bboxes = []
        face, confidence = cv.detect_face(frame)
        for idx, f in enumerate(face):
            bbox=(f[0],f[1],(f[2]-f[0]),(f[3]-f[1]))
            bboxes.append(bbox)
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        multiTracker = cv2.MultiTracker_create()
        for bbox in bboxes:
            multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
      p1 = (int(newbox[0]), int(newbox[1]))
      p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
      cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)

    # show frame
    cv2.imshow('MultiTracker', frame)
    output_movie.write(frame)
    if cv2.waitKey(2) == ord('Q'):
        break

# All done!
end=time.time()-start
print("tiempo{}".format(end))
input_movie.release()
cv2.destroyAllWindows()
