#!/home/VICOMTECH/msanz/.virtualenvs/tracking

import tensorflow as tf
import cv2
import time
import argparse
import numpy as np
import posenet
import math

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()



def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture("data/Dabadaba/Cam_1/2_1.mp4")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        len_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('data/daba_out/pose_detection_2_1.avi', fourcc, 29, (frame_width, frame_height))

        start = time.time()
        frame_count = 0
        
        while True:
            if frame_count==len_video-1:
                break
            res, img = cap.read()
            if not res:
            	break
            input_image, display_image, output_scale = posenet.process_input(img, scale_factor=args.scale_factor, output_stride=output_stride)
            #input_image, display_image, output_scale = posenet.read_cap( cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=50,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            Keypoints_face_coords=[]
            Keypoints_face_scores=[]
            for d in keypoint_coords:
                Keypoints_face_coords.append(d[0:5])
            for d in keypoint_scores:
                Keypoints_face_scores.append(d[0:5])

            Keypoints_eyes_coords=[]
            Keypoints_eyes_scores=[]
            for d in keypoint_coords:
                Keypoints_eyes_coords.append(d[1:3])
            for d in keypoint_scores:
                Keypoints_eyes_scores.append(d[1:3])

            Keypoints_half_eyes_coords = []
            Keypoints_half_eyes_scores = []

            for d in Keypoints_eyes_coords:
                Keypoints_half_eyes_coords.append(np.mean(d,axis=0))
            for d in Keypoints_eyes_scores:
                Keypoints_half_eyes_scores.append(np.mean(d,axis=0))

            Keypoints_angle_coords = []
            Keypoints_angle_scores = []
            angle = 0
            for d in Keypoints_eyes_coords:
                angle = math.atan2(d[1][1]-d[0][1], d[1][0]-d[0][0])
                Keypoints_angle_coords.append(angle)
            for d in Keypoints_eyes_scores:
                Keypoints_angle_scores.append(np.mean(d,axis=0))

            Keypoints_direction=[]
            angle_2=0
            
            for ii in range(0,len(keypoint_coords)-1): 
                if Keypoints_eyes_coords[ii][0][0] == 0 and Keypoints_eyes_coords[ii][0][1] == 0 and Keypoints_eyes_coords[ii][1][0] == 0 and Keypoints_eyes_coords[ii][1][1] == 0:
                    Keypoints_direction.append("no hay ojos") #rectoº
                    continue
                elif Keypoints_eyes_coords[ii][0][0] == 0 and Keypoints_eyes_coords[ii][0][1] == 0:
                    Keypoints_direction.append("ojo izquierdo (Derecha)") #rectoº
                    continue
                elif Keypoints_eyes_coords[ii][1][0] == 0 and Keypoints_eyes_coords[ii][1][1] == 0:
                    Keypoints_direction.append("ojo derecho (Izquierda)") #rectoº
                    continue

                if Keypoints_face_coords[ii][0][1] == 0 and Keypoints_face_coords[ii][0][0] == 0 and Keypoints_half_eyes_coords[ii][0] == 0 and Keypoints_half_eyes_coords[ii][1] == 0:
                    Keypoints_direction.append("fallo") #rectoº
                    continue
                angle_2=math.atan2(Keypoints_face_coords[ii][0][1]-Keypoints_half_eyes_coords[ii][1], Keypoints_face_coords[ii][0][0]-Keypoints_half_eyes_coords[ii][0])
                
                if abs(angle_2) < 0.261799:
                    Keypoints_direction.append("recto") #rectoº
                elif angle_2 > 0.261799:
                    Keypoints_direction.append("derecha") #derecha
                elif angle_2 < -0.261799:
                    Keypoints_direction.append("izquierda") #izq


            Keypoints_att_show_coords=[]
            Keypoints_att_show_scores=[]
            a=[]
            b=[]
            for ii in range(0,len(keypoint_coords)-1): 
                a.append(Keypoints_face_coords[ii][1])
                a.append(Keypoints_face_coords[ii][2])
                a.append(Keypoints_face_coords[ii][0])
                a.append(Keypoints_half_eyes_coords[ii])
                Keypoints_att_show_coords.append(np.asarray(a))

            for ii in range(0,len(keypoint_coords)-1): 
                b.append(Keypoints_face_scores[ii][1])
                b.append(Keypoints_face_scores[ii][2])
                b.append(Keypoints_face_scores[ii][0])
                b.append(Keypoints_half_eyes_scores[ii])
                Keypoints_att_show_scores.append(np.asarray(b))

            Keypoints_right_arm_coords=[]
            Keypoints_right_arm_scores=[]
            a=[]
            b=[]

            for d in keypoint_coords:
                a.append(d[6])
                a.append(d[8])
                a.append(d[10])
                Keypoints_right_arm_coords.append(np.asarray(a))
            for d in keypoint_scores:
                b.append(d[6])
                b.append(d[8])
                b.append(d[10])
                Keypoints_right_arm_scores.append(np.asarray(b))
            
            Keypoints_left_arm_coords=[]
            Keypoints_left_arm_scores=[]
            a=[]
            b=[]

            for d in keypoint_coords:
                a.append(d[5])
                a.append(d[7])
                a.append(d[9])
                Keypoints_left_arm_coords.append(np.asarray(a))
            for d in keypoint_scores:
                b.append(d[5])
                b.append(d[7])
                b.append(d[9])
                Keypoints_left_arm_scores.append(np.asarray(b))

            Keypoints_chest_coords=[]
            Keypoints_chest_scores=[]
            a=[]
            b=[]

            for d in keypoint_coords:
                a.append(d[5])
                a.append(d[6])
                a.append(d[11])
                a.append(d[12])
                Keypoints_chest_coords.append(np.asarray(a))
            for d in keypoint_scores:
                b.append(d[5])
                b.append(d[6])
                b.append(d[11])
                b.append(d[12])
                Keypoints_chest_scores.append(np.asarray(b))
            
            mean_points_face_coords = []
            mean_points_face_scores = []
            mean_points_chest_coords = []
            mean_points_chest_scores = []
            mean_points_right_arm_coords = []
            mean_points_right_arm_scores = []
            mean_points_left_arm_coords = []
            mean_points_left_arm_scores = []

            for d in Keypoints_face_coords:
                mean_points_face_coords.append(np.mean(d,axis=0))
            for d in Keypoints_face_scores:
                mean_points_face_scores.append(np.mean(d))

            for d in Keypoints_chest_coords:
                mean_points_chest_coords.append(np.mean(d,axis=0))
            for d in Keypoints_chest_scores:
                mean_points_chest_scores.append(np.mean(d))

            for d in Keypoints_right_arm_coords:
                mean_points_right_arm_coords.append(np.mean(d,axis=0))
            for d in Keypoints_right_arm_scores:
                mean_points_right_arm_scores.append(np.mean(d))

            for d in Keypoints_left_arm_coords:
                mean_points_left_arm_coords.append(np.mean(d,axis=0))
            for d in Keypoints_left_arm_scores:
                mean_points_left_arm_scores.append(np.mean(d))
            
            total_means_coords = []
            means_prov_coords = []
            for ii,d in enumerate(pose_scores):
                means_prov_coords.append(mean_points_face_coords[ii])
                #means_prov_coords.append(mean_points_chest_coords[ii])
                #means_prov_coords.append(mean_points_right_arm_coords[ii])
                #means_prov_coords.append(mean_points_left_arm_coords[ii])
                total_means_coords.append(np.asarray(means_prov_coords))

            total_means_scores = []
            means_prov_scores = []
            for ii,d in enumerate(pose_scores):
                means_prov_scores.append(mean_points_face_scores[ii])
                #means_prov_scores.append(mean_points_chest_scores[ii])
                #means_prov_scores.append(mean_points_right_arm_scores[ii])
                #means_prov_scores.append(mean_points_left_arm_scores[ii])
                total_means_scores.append(np.asarray(means_prov_scores))
            
            Keypoints_face_scores=np.asarray(Keypoints_face_scores)
            Keypoints_face_coords=np.asarray(Keypoints_face_coords)

            Keypoints_right_arm_coords=np.asarray(Keypoints_right_arm_coords)
            Keypoints_right_arm_scores=np.asarray(Keypoints_right_arm_scores)

            Keypoints_left_arm_coords=np.asarray(Keypoints_left_arm_coords)
            Keypoints_left_arm_scores=np.asarray(Keypoints_left_arm_scores)

            total_means_coords=np.asarray(total_means_coords)
            total_means_scores=np.asarray(total_means_scores)

            Keypoints_att_show_coords=np.asarray(Keypoints_att_show_coords)
            Keypoints_att_show_scores=np.asarray(Keypoints_att_show_scores)

            overlay_image_keypoints = posenet.draw_keypoints(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.15)

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.15)

            overlay_image_face = posenet.draw_face(
                display_image, pose_scores, Keypoints_face_scores, Keypoints_face_coords,
                min_pose_score=0.15, min_part_score=0.15)

            overlay_image_right_arm = posenet.draw_arm_right(
                display_image, pose_scores, Keypoints_right_arm_scores, Keypoints_right_arm_coords,
                min_pose_score=0.15, min_part_score=0.15)

            overlay_image_left_arm = posenet.draw_arm_left(
                display_image, pose_scores, Keypoints_left_arm_scores, Keypoints_left_arm_coords,
                min_pose_score=0.15, min_part_score=0.15)

            overlay_image_chest = posenet.draw_chest(
                display_image, pose_scores, Keypoints_chest_scores, Keypoints_chest_coords,
                min_pose_score=0.15, min_part_score=0.15)

            overlay_image_means = posenet.draw_means(
                display_image, pose_scores, total_means_scores,total_means_coords,
                min_pose_score=0.15, min_part_score=0.15)

            overlay_image_att = posenet.draw_att(
                display_image, pose_scores, Keypoints_att_show_scores,Keypoints_att_show_coords,
                min_pose_score=0.15, min_part_score=0.15)

            overlay_image_skeleton = posenet.draw_skeleton(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.15)
            #cv2.imshow('posenet_body', overlay_image)
            #cv2.imshow('posenet_face', overlay_image_face)
            #cv2.imshow('posenet_keypoints', overlay_image_keypoints)
            #cv2.imshow('posenet_skeleton', overlay_image_skeleton)
            #cv2.imshow('posenet_right_arm', overlay_image_right_arm)
            #cv2.imshow('posenet_left_arm', overlay_image_left_arm)
            #cv2.imshow('posenet_means', overlay_image_means)
            #cv2.imshow('posenet_chest', overlay_image_chest)
            cv2.imshow('posenet_att', overlay_image_att)
            #output_movie.write(overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))
        print('time: ', (time.time() - start))


if __name__ == "__main__":
    main()
