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

The important files are:

- image_divider.py: This file performs two actions, the division of images and the preprocessing with the brightness, contrast and gamma correction. The important constants that can be tunned are gamma (the deggree for the correction of the image in the gamma), contrast (the deggree for the correction of the image in the contrast), color (Selection of color image, 1, or grayscale, 0) and img (selecting the image or frame to read).
- pose_detection.py: This file performs the detection of different joints, being them eyes, ears, nose, shoulders, elbows, hands, hips, kness and feet. With those joints some different compositionts had been done:
    + Only points: Only the points of the joints detected are drawn, it can be seen in the overlay_image_keypoints image.
    + Face: Only the points of the points on the face are drawn, it can be seen in the overlay_image_keypoints image.
    + Only skeleton: Only the skeleton connecting the joints detected is drawn, it can be seen in the overlay_image_skeleton image.
    + Points and skeleton: Both the skeleton connecting the joints detected and the joints are drawn, it can be seen in the overlay_image image.
    + Arms: In this case only the joints in the arms are drawn, being them in the images overlay_image_right_arm and overlay_image_left_arm for each one of the arms.
    + Mean face point: In this case the face position is calculated by averaging the position of all the points in the face. It is in the image overlay_image_means and can be used with more points than the face if wanted.
    + Chest: In this case only the joints in the chest are drawn, being them in the image overlay_image_chest.
    + Attention points: In this case the points used for the attention detection, eyes, midle of eyes and nose. This can be seen in the image overlay_image_att.
- pose_detection_seguir.py: This file performs the same action as the file pose_detection.py but it adds the functions for tracking, the important constans are the same as before but also the number of frames between two detection frames called n_frames_to_detect, it is also important the variable trackingQuality_threshold, which determine the quality to following trade-off and the variable sec that deteermines the number of seconds to show the movement.
- prueba_face_detection_image.py: This file performs the face detection in an image, the most important variables are f, being the file name of the faces to analyse and all the other variables in the file faces.txt
- prueba_face_detection_videos.py: This file performs the same action as the previous one with the difference of doing to all the frames in a video. The important variables are the input movie.
- prueba_object_detection_image.py: This file performs the same action as the face detection image but with objects instead of persons.
- prueba_object_detection_video.py: This file performs the same action as the face detection video but with objects instead of persons.
- track_face_video_2.py: This file performs This file implements the tracking to the video face detection, being the new important variables the trackingQuality_threshold, control of the tradeof between the tracking quality and speed, n_frames_to_detect, number of frames between two detection frames.
- track_face_video_2_seguir.py: This file performs the same action as track_face_video_2 adding the movement following, being them the representation of the last points with three colors depending on the recent of the mesurement.
- track_object_video_2.py: This file performs the same as the tracking of faces to the object.
- track_pose_detection.py: This file performs the same as the tracking to the people joints.

in the case of wanting to run all the files in cascade it can be done with the following command sh run_all.sh

## Instalation

The instalation for this needs the following:

- Python 3 environment: mkvirtualenv -a $(pwd) createdenv
- Tensorflow: pip install tensorflow or pip install tensorflow-gpu for usage in gpu
- OpenCV: pip install opencv-python==3.4.5.2
- scipy: pip install scipy
- yaml: pip install pyyaml
- cvlib: pip install cvlib
- dlib: pip install dlib
- face_recognition: pip install face_recognition
- imageai: Download [the image ai from the link](https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl)and in that folder pip install imageai-2.0.2-py3-none-any.whl
- matplotlib: pip install matplotlib

## Test done

All the test done are runned with two videos, one with a low quantity of people and good light conditions, and the other with bad conditions and high number of people. The test done has been:

- Person detection: In this test it has been seen that the persons detection is very dependable on the conditions and the distance to the camera.

- Face detection: In this case the results are very similar to the previous test but reduce the computation time as it search for a smaller part.

- Chest detection: In this case in the good video the results are very similar but in the other the results are worse as a the chest is more difficult to see in that conditions.

- Attention detection: In this test the points for the attention detection is shown but the most important part is the file written with the attention direction of each tracked person at each moment.

- Movement tracking: In this test the position across the time of the people is stored in a variable, which is then used to show the path taken for that person.

## Improvements

The main improvement to implement is and improvement in the detetion that although a preprocessing is implemented.

Other improvement is the solution of the bug where a person is not detected and then is detected again the movement counter is reset.

# Wi-Fi

## Parts

- Server

- Scanner

## Metrics

## Ussage

## Instalation

## Test done

## Improvements

# Hybrid

This part has not been done yet. This part is based on the results of both method to perform changes to the techniques and improve their results.
