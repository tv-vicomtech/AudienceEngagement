import face_recognition
import cv2
import cvlib as cv
import time


input_movie = cv2.VideoCapture("data/test_videos/hamilton_clip.mp4")

if not input_movie.isOpened():
    print("Could not open video file")
    exit()

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(input_movie.get(3))
frame_height = int(input_movie.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie_fd = cv2.VideoWriter('data/videos_face_det/output_fd.avi', fourcc, 29.97, (frame_width, frame_height))

output_movie_cv = cv2.VideoWriter('data/videos_face_det/output_cv.avi', fourcc, 29.97, (frame_width, frame_height))

frame_number = 0
elapsed_time_fd = 0
elapsed_time_cv = 0
while input_movie.isOpened():
    # Grab a single frame of video
    ret, frame = input_movie.read()
    if not ret:
        print("Could not read frame")
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    start_fd = time.time()
    face_locations = face_recognition.face_locations(rgb_frame)
    elapsed_time_fd += (time.time() - start_fd)
    for (top, right, bottom, left) in face_locations:

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Write the resulting image to the output video file
    print("Writing frame for face_detection {} / {}".format(frame_number, length))
    output_movie_fd.write(frame)

    #ret, frame = input_movie.read()

    start_cv = time.time()
    face, confidence = cv.detect_face(frame)
    elapsed_time_cv += (time.time() - start_cv)

    for idx, f in enumerate(face):
        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

    print("Writing frame for cvlib {} / {}".format(frame_number, length))
    output_movie_cv.write(frame)
    frame_number += 1

print("time cv{}\ntime fd{}".format(elapsed_time_cv,elapsed_time_fd))
# All done!
input_movie.release()
cv2.destroyAllWindows()
