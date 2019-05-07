import cv2
import glob
import dlib
import argparse
import time
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
##
import cvlib as cv
import cv2
import os 

##
import cv2
import dlib
import argparse

f=open("benchmarks/faces.txt","r")
filenames = [imag for imag in glob.glob("data/test_images/*")]

filenames.sort() # ADD THIS LINE
n=0
images = []
##face detection with face_recognition
for imag in filenames:
    print('\n\nImagen nueva')
    a=int(f.readline())
    image = face_recognition.load_image_file(imag)

    start = time.time()
    face_locations = face_recognition.face_locations(image)
    elapsed_time = time.time() - start
    b=len(face_locations)
    if b>a:
        acc=100*(a/b)
    else:
        acc=100*(b/a)

    print("Time for the face detection with face_reconognition: {}, with {} faces detected".format(elapsed_time, b))
    print("The image had {} faces, so it has an accuracy of {}%".format(a, acc))
    imagen = cv2.imread(imag)
    for (top, right, bottom, left) in face_locations:
    
            # Draw a box around 
        cv2.rectangle(imagen, (left, top), (right, bottom), (0,255,0), 2)
    
    cv2.imwrite('data/images_faces_det/image{}_face_recognition.jpg'.format(n),imagen)

    ##face detection with cvlib
    
    
    # read input image
    
    image = cv2.imread(imag)
    if image is None:
        print("Could not read input image")
        exit()

    # apply face detection
    start = time.time()
    faces, confidences = cv.detect_face(image)
    elapsed_time = time.time() - start
    b=len(faces)
    if b>a:
        acc=100*(a/b)
    else:
        acc=100*(b/a)

    print("Time for the face detection with cvlib: {} with {} faces detected".format(elapsed_time, b))
    print("The image had {} faces, so it has an accuracy of {}%".format(a, acc))
    
    for face in faces:

        # Draw a box around the face using the Pillow module
        cv2.rectangle(image,(face[0], face[1]), (face[2], face[3]), (0, 0, 255), 2)

    cv2.imwrite('data/images_faces_det/image{}_cvlib.jpg'.format(n),image)

    n+=1

