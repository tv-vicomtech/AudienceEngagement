import tensorflow as tf
import cv2
import time
import argparse
import dlib
import glob
import cvlib as cv
import threading
import time
import os 
import numpy as np
import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()

trackingQuality_threshold = 9
n_frames_to_detect = 10
min_confidence = 0.55
sec=5
min_points = 9 #5 for face 9 face and arms11 for upper body >17 for whole body 
face_points = 5 
arms_points =  10

def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture("data/test_videos/dinner.mp4")
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        len_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('data/track_video/pose_detection.avi', fourcc, 29.97, (args.cam_width, args.cam_height))

        listofcenters = []
        centers = []
        max_displacement = []
        max_movement_single=[]

        frame_number = 0
        currentFaceID = 0
        cont=0

        rectangleColor = (0,0,255)
        faceTrackers = {}

        start = time.time()
        frame_count = 0
        while True:
            if int(time.time()-start)> ((cont+1)*5):
                cont+=1
                print("Total movement for the past {} seconds:".format(cont*5))
                for item in max_displacement:
                    print(item)
                print("Maximun movement for the past {} seconds:".format(cont*5))
                for item_2 in max_movement_single:
                    print(item_2)

            if frame_count==len_video:
                break
            res, img = cap.read()
            if not res:
                break
            input_image, display_image, output_scale = posenet.process_input(img, scale_factor=args.scale_factor, output_stride=output_stride)
            #input_image, display_image, output_scale = posenet.read_cap( cap, scale_factor=args.scale_factor, output_stride=output_stride)
            frame_count += 1

            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[ fid ].update( overlay_image )

                if trackingQuality < trackingQuality_threshold:
                    fidsToDelete.append( fid )

            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop( fid , None )


            if (frame_number % n_frames_to_detect) == 0:

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    model_outputs,
                    feed_dict={'image:0': input_image}
                )

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses( heatmaps_result.squeeze(axis=0), offsets_result.squeeze(axis=0), displacement_fwd_result.squeeze(axis=0), displacement_bwd_result.squeeze(axis=0), output_stride=output_stride, max_pose_detections=10, min_pose_score=0.15)

                keypoint_coords *= output_scale

                overlay_image = posenet.draw_skel_and_kp( display_image, pose_scores, keypoint_scores, keypoint_coords, min_pose_score=0.15, min_part_score=0.1)
                
                for idx,esq in enumerate(keypoint_coords):
                    x_coords = []
                    y_coords = []

                    right_arm_x_coords = []
                    right_arm_y_coords = []

                    left_arm_x_coords = []
                    left_arm_y_coords = []

                    for ii in range(0,min(min_points,len(esq)-1)):
                        if esq[ii][0]!=0 or esq[ii][1]!=0:
                            if ii < face_points:
                                x_coords.append(esq[ii][0])
                                y_coords.append(esq[ii][1]) 
                            elif ii < arms_points:
                                if ii % 2:
                                    left_arm_x_coords.append(esq[ii][0])
                                    left_arm_y_coords.append(esq[ii][1])
                                else:
                                    right_arm_x_coords.append(esq[ii][0])
                                    right_arm_y_coords.append(esq[ii][1])

                    
                    if len(x_coords)!=0:
                        x_min = np.min(x_coords)-20
                        y_min = np.min(y_coords)-20
                        x_max = np.max(x_coords)+20
                        y_max = np.max(y_coords)+20
                        x= x_min
                        y= y_min
                        w= x_max - x_min
                        h= y_max - y_min
                        x_bar= x + 0.5 * w
                        y_bar= y + 0.5 * h
                        #print(esq,x_min,x_max,y_max,y_min,"\n\n\n")
                        #cv2.rectangle(overlay_image, (int(y_min), int(x_min)),(int(y_max) ,int(x_max)), (0,255,0) ,2)
                        #cv2.circle(overlay_image, (int(y_bar), int(x_bar)),5, (0,255,0) ,2)
                        #cv2.circle(overlay_image, (int(y_max), int(x_max)),5, (0,255,0) ,2)
                        

                        matchedFid = None
                        #
                    
                        for fid in faceTrackers.keys():
                            tracked_position = faceTrackers[fid].get_position()

                            t_x= int(tracked_position.left())
                            t_y= int(tracked_position.top())
                            t_w= int(tracked_position.width())
                            t_h= int(tracked_position.height())
                            t_x_bar= t_x + 0.5 * t_w
                            t_y_bar= t_y + 0.5 * t_h
                            #cv2.circle(overlay_image, (int(t_x_bar), int(t_y_bar)),5, (255,0,0) ,2)
                            #cv2.circle(overlay_image, (int(t_x_bar), int(t_x_bar)),5, (255,0,0) ,2)
                            if ( ( t_y <= x_bar   <= (t_y + t_h)) and 
                                 ( t_x <= y_bar   <= (t_x + t_w)) and 
                                 ( x   <= t_y_bar <= (x   + w  )) and 
                                 ( y   <= t_x_bar <= (y   + h  ))):
                                matchedFid = fid
                                centers=listofcenters[fid]
                                centers=[(int(y_bar),int(x_bar))]+centers
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
                            tracker.start_track(overlay_image,dlib.rectangle(int(y_min), int(x_min), int(y_max) , int(x_max)))
                            faceTrackers[ currentFaceID ] = tracker
                            currentFaceID += 1
                            centers=[]
                            centers=[(int(y_bar),int(x_bar))]+centers
                            listofcenters.append(centers)
                            max_displacement.append(0)
                            max_movement_single.append(0)

                    if len(left_arm_x_coords)!=0:
                        x_min = np.min(left_arm_x_coords)-20
                        y_min = np.min(left_arm_y_coords)-20
                        x_max = np.max(left_arm_x_coords)+20
                        y_max = np.max(left_arm_y_coords)+20
                        x= x_min
                        y= y_min
                        w= x_max - x_min
                        h= y_max - y_min
                        x_bar= x + 0.5 * w
                        y_bar= y + 0.5 * h
                        #print(esq,x_min,x_max,y_max,y_min,"\n\n\n")
                        #cv2.rectangle(overlay_image, (int(y_min), int(x_min)),(int(y_max) ,int(x_max)), (0,255,0) ,2)
                        #cv2.circle(overlay_image, (int(y_bar), int(x_bar)),5, (0,255,0) ,2)
                        #cv2.circle(overlay_image, (int(y_max), int(x_max)),5, (0,255,0) ,2)
                        

                        matchedFid = None
                        #
                    
                        for fid in faceTrackers.keys():
                            tracked_position = faceTrackers[fid].get_position()

                            t_x= int(tracked_position.left())
                            t_y= int(tracked_position.top())
                            t_w= int(tracked_position.width())
                            t_h= int(tracked_position.height())
                            t_x_bar= t_x + 0.5 * t_w
                            t_y_bar= t_y + 0.5 * t_h
                            #cv2.circle(overlay_image, (int(t_x_bar), int(t_y_bar)),5, (255,0,0) ,2)
                            #cv2.circle(overlay_image, (int(t_x_bar), int(t_x_bar)),5, (255,0,0) ,2)
                            if ( ( t_y <= x_bar   <= (t_y + t_h)) and 
                                 ( t_x <= y_bar   <= (t_x + t_w)) and 
                                 ( x   <= t_y_bar <= (x   + w  )) and 
                                 ( y   <= t_x_bar <= (y   + h  ))):
                                matchedFid = fid
                                centers=listofcenters[fid]
                                centers=[(int(y_bar),int(x_bar))]+centers
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
                            tracker.start_track(overlay_image,dlib.rectangle(int(y_min), int(x_min), int(y_max) , int(x_max)))
                            faceTrackers[ currentFaceID ] = tracker
                            currentFaceID += 1
                            centers=[]
                            centers=[(int(y_bar),int(x_bar))]+centers
                            listofcenters.append(centers)
                            max_displacement.append(0)
                            max_movement_single.append(0)

                    if len(right_arm_x_coords)!=0:
                        x_min = np.min(right_arm_x_coords)-20
                        y_min = np.min(right_arm_y_coords)-20
                        x_max = np.max(right_arm_x_coords)+20
                        y_max = np.max(right_arm_y_coords)+20
                        x= x_min
                        y= y_min
                        w= x_max - x_min
                        h= y_max - y_min
                        x_bar= x + 0.5 * w
                        y_bar= y + 0.5 * h
                        #print(esq,x_min,x_max,y_max,y_min,"\n\n\n")
                        #cv2.rectangle(overlay_image, (int(y_min), int(x_min)),(int(y_max) ,int(x_max)), (0,255,0) ,2)
                        #cv2.circle(overlay_image, (int(y_bar), int(x_bar)),5, (0,255,0) ,2)
                        #cv2.circle(overlay_image, (int(y_max), int(x_max)),5, (0,255,0) ,2)
                        

                        matchedFid = None
                        #
                    
                        for fid in faceTrackers.keys():
                            tracked_position = faceTrackers[fid].get_position()

                            t_x= int(tracked_position.left())
                            t_y= int(tracked_position.top())
                            t_w= int(tracked_position.width())
                            t_h= int(tracked_position.height())
                            t_x_bar= t_x + 0.5 * t_w
                            t_y_bar= t_y + 0.5 * t_h
                            #cv2.circle(overlay_image, (int(t_x_bar), int(t_y_bar)),5, (255,0,0) ,2)
                            #cv2.circle(overlay_image, (int(t_x_bar), int(t_x_bar)),5, (255,0,0) ,2)
                            if ( ( t_y <= x_bar   <= (t_y + t_h)) and 
                                 ( t_x <= y_bar   <= (t_x + t_w)) and 
                                 ( x   <= t_y_bar <= (x   + w  )) and 
                                 ( y   <= t_x_bar <= (y   + h  ))):
                                matchedFid = fid
                                centers=listofcenters[fid]
                                centers=[(int(y_bar),int(x_bar))]+centers
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
                            tracker.start_track(overlay_image,dlib.rectangle(int(y_min), int(x_min), int(y_max) , int(x_max)))
                            faceTrackers[ currentFaceID ] = tracker
                            currentFaceID += 1
                            centers=[]
                            centers=[(int(y_bar),int(x_bar))]+centers
                            listofcenters.append(centers)
                            max_displacement.append(0)
                            max_movement_single.append(0)


            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                t_x_bar = int(t_x + 0.5 * t_w)
                t_y_bar = int(t_y + 0.5 * t_h)
                centers=listofcenters[fid]
                centers=[(t_x_bar,t_y_bar)]+centers
                listofcenters[fid]=centers
                max_distance=max_displacement[fid]
                for (x,y) in centers:
                    distance=abs((pow(x_bar,2)+pow(y_bar,2))-(pow(t_x_bar,2)+pow(t_y_bar,2)))
                    max_displacement[fid]+=distance
                    if distance > max_movement_single[fid]:
                        max_movement_single[fid]=distance
                cv2.rectangle(overlay_image, (int(t_x), int(t_y)),(int(t_x + t_w) ,int(t_y +t_h)), (0,255,0) ,2)
                cv2.circle(overlay_image, (int(t_x_bar), int(t_y_bar)),5, (0,255,0) ,2)


            # List to hold x values.
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
                        cv2.line(overlay_image,(x_1,y_1),(x_2,y_2),(255,0,0),4)
                    elif stop < 40:
                        cv2.line(overlay_image,(x_1,y_1),(x_2,y_2),(0,255,0),2)
                    elif stop < 100:
                        cv2.line(overlay_image,(x_1,y_1),(x_2,y_2),(0,0,255),1)
                    stop+=1

            cv2.imshow('posenet', overlay_image)
            output_movie.write(overlay_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))
        print('time: ', (time.time() - start))
        print("Total movement for the past {} seconds:".format(end))
        for item in max_displacement:
            print(item)
        print("Maximun movement for the past {} seconds:".format(end))
        for item_2 in max_movement_single:
            print(item_2)

if __name__ == "__main__":
    main()
