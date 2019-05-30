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
- imageai: Download [the image ai from the link](https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl) and in that folder pip install imageai-2.0.2-py3-none-any.whl
- matplotlib: pip install matplotlib

## Test done

All the test done are runned with two videos, one with a low quantity of people and good light conditions, and the other with bad conditions and high number of people. The test done has been:

- Person detection: In this test it has been seen that the persons detection is very dependable on the conditions and the distance to the camera.

- Face detection: In this case the results are very similar to the previous test but reduce the computation time as it search for a smaller part.

- Chest detection: In this case in the good video the results are very similar but in the other the results are worse as a the chest is more difficult to see in that conditions.

- Attention detection: In this test the points for the attention detection is shown but the most important part is the file written with the attention direction of each tracked person at each moment.

- Movement tracking: In this test the position across the time of the people is stored in a variable, which is then used to show the path taken for that person.

# Wi-Fi

## Parts

This technique is composed by two parts:

- Server: This part is in charge of storing the information and computing the zones. In this case is done in a computer

- Scanner: This part is in charge of dettecting all the devices and sending it to the server. In this case is done in severall raspberry Pi devices with wifi dongles.

## Metrics

In this case the metrics obtained are:

- Position of people: This is known by ussing knowing the zones and seeing the devices in that zone, this is stored in a vector called zones.
- Number of people: This is known by seing the lines of the devices.json file and the vector to_find.

## Ussage

### Server

The server can be used with or without docker:

#### With Docker

To start the docker with the Find3 server this commands needs to be input:

sudo docker start -p 1884:1883 -p 8005:8003 -v /home/$USER/FIND_DATA:/data -e MQTT_ADMIN=ADMIN -e MQTT_PASS=PASSWORD -e MQTT_SERVER='localhost:1883' -e MQTT_EXTERNAL=192.168.35.101 -e MQTT_PORT=1884 --name wifi -d -t schollz/find3

#### Without Docker
In order to use this part it needs to be installed in the server computer and in the scanner devices(Raspberry Pi), it is needed to be first run the server by going to 

$ cd $GOPATH/src/github.com/user/wifi_server/server/ai
$ make

and in another teminal

$ cd $GOPATH/src/github.com/user/wifi_server/server/main
$ go build -v
$ ./main -port 8005 

With this the server is up an running, the first group is not necesary but solves some errors. 

###Scanner

The scanner devices are used by the commands

$ wifi_scan -i YOURINTERFACE -monitor-mode

this command is to allow the monitor mode in the interface

$ wifi_scan -i YOURINTERFACE -device YOURDEVICE -family YOURFAMILY -server http://IP:PORT -scantime 300 -bluetooth -forever -passive -no-modify &

This command allow the scanning and has different variables:

1. i: this is followed by the interface of the wifi used as "wlan0".
2. device: This is followed by the name of the scanning device for the localization as "Rasberry_PI_2B".
3. family: This is followed by the family name, a way of relating all the measurements fron all the devices as "Localization"
4. server: This is followed by the URL or IP of the server, if the IP is given the port of connection is needed, as "http://192.168.10.30:8005".
5. scantime: This is followed by the maximum scanning time in seconds, as "300".
6. forever: This tag make it to run in an infinity loop instead of only one time
7. passive: This tag indicates that the scanning is passive and not active 
8. no-modify: This tag is used to improve performance by not changing the interface mode to monitor in every loop.
9. bluetooth: this tag indicates that it need to monitor both wifi and bluettoth signals.

## Instalation

### Scanner

The scanner can be installed as:

The instalation for this part needs Golang which can be download from [their web](https://golang.org/), go to the downloads folder and extract it to /usr/local, export the path as:

$ export PATH=$PATH:/usr/local/go/bin

and the gopath to the path inside the go folder created

$ export GOPATH=$(go env GOPATH)

Create a folder for the wifi_detection:

$ mkdir $GOPATH/src/github.com/user/WIFI_detection

and inside of it include all the files from the wifi_scanner folder. The file mac_vendors.txt has to be put in a known folder and write that path in the reverse.go file line 38.

Go inside the folder and build the code

$ cd $GOPATH/src/github.com/user/WIFI_detection
$ go install

Move the binary file to a accesible path with: 

$ sudo mv $GOPATH/bin/wifi_scan /usr/local/bin

and it can be run with the command:

$ wifi_scan -i YOURINTERFACE -monitor-mode

$ wifi_scan -i YOURINTERFACE -device YOURDEVICE -family YOURFAMILY -server http://IP:PORT -scantime 300 -forever -passive -no-modify &

### Server

The server can be installed with or without docker

#### With Docker

In this case the instalation lose the performance improvements and some of the changes, but gains in reliability and it is easier to use and install. First docker must be installed as:

$ curl -sSL https://get.docker.com | sh

The repository can be download as:

$ docker pull schollz/find3

#### Without docker

Install the C compiler and the go compiler, go as before and C with:

$ sudo apt-get install g++

Install mosquitto:

$ sudo apt-get install mosquitto-clients mosquitto

Create a folder for the wifi_detection:

$ mkdir $GOPATH/src/github.com/user/WIFI_detection

and inside of it include all the files from the wifi folder, go inside the folder and build the code

$ go build -v
$ ./main -port 8005

## Test done

The test done are:

1. Devices across the room: In this test two scanning positions were used and compared, while the tracked devices were placed in different parts of the room at the same time.
2. Devices in the same part of the room: In this test two scanning positions were used and compared, while the tracked devices were placed in the same part of the room at the same time.
3. Devices moving: In this test two scanning positions were used and compared, while the tracked devices were moving in a determined path ath the same time.
4. Devices covered by a body: In this case only the best position for the scannig devices were used, a human body were placed between the scanner and tracked device.
5. Devices being used: In this case only the best position for the scannig devices were used, while this test was done the devices were hold as it were a normal usage.

# Hybrid

This part is based on the results of both method to perform changes to the techniques and improve their results.
