from imageai.Detection import VideoObjectDetection
import os
import time
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2



input_movie = cv2.VideoCapture('data/test_videos/hamilton_clip.mp4')
if not input_movie.isOpened():
    print("Could not open webcam")
    exit()

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(input_movie.get(3))
frame_height = int(input_movie.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie_cv = cv2.VideoWriter('data/test_videos/output_cv.avi', fourcc, 29.97, (frame_width, frame_height))

execution_path = os.getcwd()

elapsed_time_cv = 0
frame_number=0

while input_movie.isOpened():

    # read frame from webcam 
    status, frame = input_movie.read()
    print("Writing frame for face_detection {} / {}".format(frame_number, length))
    if not status:
        print("Could not read frame")
        break

    # apply object detection
    start_cv = time.time()
    bbox, label, conf = cv.detect_common_objects(frame)
    elapsed_time_cv += (time.time() - start_cv)

    #print(bbox, label, conf)

    # draw bounding box over detected objects
    out = draw_bbox(frame, bbox, label, conf)

    output_movie_cv.write(frame)
    frame_number+=1

print("time spent in conversion: {}".format(elapsed_time_cv))
input_movie.release()

detector5 = VideoObjectDetection()
detector5.setModelTypeAsYOLOv3()
detector5.setModelPath( os.path.join(execution_path , "model_data/yolo.h5"))
detector5.loadModel()

custom_objects = detector5.CustomObjects(person=True)

start=time.time()
video_path = detector5.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join( execution_path, "data/test_videos/hamilton_clip.mp4"), output_file_path=os.path.join(execution_path, "data/test_videos/obj_det_normal"), frames_per_second=29)
eplased_time=time.time()-start
print("time spent in conversion: {}".format(eplased_time))
print(video_path)

detector5 = VideoObjectDetection()
detector5.setModelTypeAsYOLOv3()
detector5.setModelPath( os.path.join(execution_path , "model_data/yolo.h5"))
detector5.loadModel(detection_speed="fast")

custom_objects = detector5.CustomObjects(person=True)

start=time.time()
video_path = detector5.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join( execution_path, "data/test_videos/hamilton_clip.mp4"), output_file_path=os.path.join(execution_path, "data/test_videos/obj_det_fast"), frames_per_second=29)
eplased_time=time.time()-start
print("time spent in conversion: {}".format(eplased_time))
print(video_path)

detector5 = VideoObjectDetection()
detector5.setModelTypeAsYOLOv3()
detector5.setModelPath( os.path.join(execution_path , "model_data/yolo.h5"))
detector5.loadModel(detection_speed="faster")

custom_objects = detector5.CustomObjects(person=True)

start=time.time()
video_path = detector5.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join( execution_path, "data/test_videos/hamilton_clip.mp4"), output_file_path=os.path.join(execution_path, "data/test_videos/obj_det_faster"), frames_per_second=29)
eplased_time=time.time()-start
print("time spent in conversion: {}".format(eplased_time))
print(video_path)

detector5 = VideoObjectDetection()
detector5.setModelTypeAsYOLOv3()
detector5.setModelPath( os.path.join(execution_path , "model_data/yolo.h5"))
detector5.loadModel(detection_speed="fastest")

custom_objects = detector5.CustomObjects(person=True)

start=time.time()
video_path = detector5.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join( execution_path, "data/test_videos/hamilton_clip.mp4"), output_file_path=os.path.join(execution_path, "data/test_videos/obj_det_fastest"), frames_per_second=29)
eplased_time=time.time()-start
print("time spent in conversion: {}".format(eplased_time))
print(video_path)

detector5 = VideoObjectDetection()
detector5.setModelTypeAsYOLOv3()
detector5.setModelPath( os.path.join(execution_path , "model_data/yolo.h5"))
detector5.loadModel(detection_speed="flash")

custom_objects = detector5.CustomObjects(person=True)

start=time.time()
video_path = detector5.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join( execution_path, "data/test_videos/hamilton_clip.mp4"), output_file_path=os.path.join(execution_path, "data/test_videos/obj_det_flash"), frames_per_second=29)
eplased_time=time.time()-start
print("time spent in conversion: {}".format(eplased_time))
print(video_path)


    
