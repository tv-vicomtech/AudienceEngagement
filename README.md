# Vicomtech engagement detection system

This repository contains a master thesis done in Vicomtech to design, implement and validate a system capable of measuring the audience engagement in a live event. This system is based on two techniques, computer vision and RSSI detection. 

This repository has a system using two techniques to have an engagement system detecter for live event system

This system has a vision technique system based on tensorflow and a wifi tracking system based on the Find3 repository. To see a more detailed information about the working principle of each one of the techniques and the PDF.

# Computer vision

This technique make use of computer vision with machine learning to process a video and make the detection of the people and engagement. For this several steps has been done, begining with the detection of people in images and then with videos. The different procces done in this part are the person detection, face detection, joint detection, movement tracking and attention detection. 

## Parts

In the part of the computer vision the division is done in different parts:

1. Detection: The detection can be done with three methods, person, face and joints. The main difference from them is what it is detected being respectivelly the person, face or several parts of the body (eyes, ears, nose, shoulders, elbows, hands, hips, knees and feet). The one that gives more information is the joint detection, but it requires more power. 

2. Tracking: The tracking is done because two reasons, reduce the computation power necesary as it allow to perform the detection in 1 out of 10 frames and maintain similar results, and the second reason is to have a movement tracking with a relation with the person who had done the moves.

3. Attention: The direction where the attention is given can be calculated with the positon of the eyes and nose.

## Metrics

The metrics of this parts are the following:


1. Position of person: This metrics is known reading the position of all the joints, this can be symplified with other vectors that only store the position of some parts such as the mean face vector, storing a mean position for the face.
2. Number of person: This metrics is known by the length of the vector where the position of the element's positions.
3. Movement of person: This metric is given by two vectors, the total movement one and the maximun movement, which shows the sum of all the distances that that person has done, if the person is not detected the movement vector is restored, and the second vector shows the maximum movement between two detection frames.
4. Attention of person: This metric is known by a vector which stores a variable with five possible values ("no eyes","error","front","left","right").

## Ussage

There is a command to run the computer vision which is symply running the python file as:

**python filename.py**

The different files are:

- image_divider.py: This file performs two actions, the division of images and the preprocessing with the brightness, contrast and gamma correction. The important constants that can be tunned are gamma (the deggree for the correction of the image in the gamma), contrast (the deggree for the correction of the image in the contrast), color (Selection of color image, 1, or grayscale, 0) and img (selecting the image or frame to read).
- pose_detection.py: This file performs the detection of different parts of the 
- pose_detection_seguir.py: This file performs the same action as the file pose_detection.py but it adds the functions for tracking, the important constans are the same as before but also the number of frames between two detection frames called n_frames_to_detect, it is also important the variable trackingQuality_threshold, which determine the quality to following trade-off and the variable sec that deteermines the number of seconds to show the movement
- pose_detection_seguir_face_arms.py: This file performs 
- prueba_face_detection_image.py: This file performs
- prueba_face_detection_videos.py: This file performs
- prueba_object_detection_image.py: This file performs
- prueba_object_detection_video.py: This file performs
- track_face_video.py: This file performs
- track_face_video_2.py: This file performs
- track_face_video_seguir.py: This file performs
- track_face_video_2_seguir.py: This file performs
- track_object_video.py: This file performs
- track_object_video_2.py: This file performs
- track_object_video_3.py: This file performs
- track_pose_detection.py: This file performs
- yolo.py: This file performs

in the case of wanting to run all the files in cascade it can be done with the following command sh run_all.sh

## Instalation

## Test done

## Improvements

# Wi-Fi

## Parts

## Metrics

## Ussage

## Instalation

## Test done

## Improvements

# Hybrid

This part has not been done yet. This part is based on the results of both method to perform changes to the techniques and improve their results.
