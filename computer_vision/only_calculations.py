import os
import cv2
import time
import dlib
import glob
import time
import math
import json
import string
import posenet
import argparse
import threading


import cvlib as cv
import numpy as np
import tensorflow as tf


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def arguments_com(args):
    relation_vector = np.array(json.loads(args.relation_zones))
    contrast_vector = np.array(json.loads(args.contrast))
    gamma_vector    = np.array(json.loads(args.gamma))

    if args.grid_size !=4 and args.grid_size !=2 and args.grid_size !=1:
        print("The grid size must be either 1,2 or 4 for better results")
        exit()
    elif args.detection_zone!=5 and args.detection_zone!=11 and args.detection_zone!=17:
        print("The detection zone must be either 5 (for face), 11 (for upper body) or 17 (for whole body)")
        exit()
    elif args.track_quality!=7 and args.track_quality!=8:
        print("The tracking quality must be either 7 or 8 for better results")
        exit()
    elif np.mean(contrast_vector)<5 or np.mean(contrast_vector)>15:
        print("The contrast must be between 5 and 15 for better results")
        exit()
    elif np.mean(gamma_vector)<2.75 or np.mean(gamma_vector)>5.0:
        print("The gamma correction factor must be between 2.75 and 5.0 for better results")
        exit()
    elif args.cam_or_file !=0 and args.cam_or_file!=1:
        print("Select the source of the video 0 for file and 1 for camera") 
        exit()
    elif args.cam_or_file ==0 and args.input_file_name=="a":
        print("Introduce a file name")
        exit()
    elif args.n_divisions != 6 and args.n_divisions !=9:
        print("number of division must be either 6 or 9",args.n_divisions)
        exit() 
    elif args.n_divisions !=len(contrast_vector) or args.n_divisions!=len(gamma_vector):
        print("Number of divisions must be equal to length of gamma and contrast vectors")
        exit()

def zeros_number_partition(args):
    number_partition=[0]*args.n_divisions
    return number_partition

def division_funct(height,width,args):
    height_fin          = height*(8/10)
    width_int           = width/3
    height_min          = height*(2/7)
    width_min           = 0


    if args.n_divisions==6:
        height_int          = (height_min + height_fin)/2
        h_up                = [int(height_int),int(height_int),int(height_int),int(height_fin),int(height_fin),int(height_fin)]
        h_down              = [int(height_min),int(height_min),int(height_min),int(height_int),int(height_int),int(height_int)]

        w_up                = [int(width_int),int(width_int*2),int(width_int*3),int(width_int),int(width_int*2),int(width_int*3)]
        w_down              = [int(width_min),int(width_int),int(width_int*2),int(width_min),int(width_int),int(width_int*2)]

        number_partition    = zeros_number_partition(args)
        zones               = [(h_down[0],h_up[0],w_down[0],w_up[0]),(h_down[1],h_up[1],w_down[1],w_up[1]),(h_down[2],h_up[2],w_down[2],w_up[2]),(h_down[3],h_up[3],w_down[3],w_up[3]),(h_down[4],h_up[4],w_down[4],w_up[4]),(h_down[5],h_up[5],w_down[5],w_up[5])]

    elif args.n_divisions==9:
        height_int          = (height_fin - height_min)/3
        h_up                = [int(height_min + height_int), int(height_min + height_int), int(height_min + height_int), int(height_min + 2*height_int), int(height_min + 2*height_int), int(height_min + 2*height_int), int(height_fin), int(height_fin), int(height_fin)]
        h_down              = [int(height_min), int(height_min), int(height_min), int(height_min + height_int), int(height_min + height_int), int(height_min + height_int), int(height_min + 2*height_int), int(height_min + 2*height_int), int(height_min + 2*height_int)]

        w_up                = [int(width_int),int(width_int*2),int(width_int*3),int(width_int),int(width_int*2),int(width_int*3),int(width_int),int(width_int*2),int(width_int*3)]
        w_down              = [int(width_min),int(width_int),int(width_int*2),int(width_min),int(width_int),int(width_int*2),int(width_min),int(width_int),int(width_int*2)]

        number_partition    = zeros_number_partition(args)
        zones               = [(h_down[0],h_up[0],w_down[0],w_up[0]),(h_down[1],h_up[1],w_down[1],w_up[1]),(h_down[2],h_up[2],w_down[2],w_up[2]),(h_down[3],h_up[3],w_down[3],w_up[3]),(h_down[4],h_up[4],w_down[4],w_up[4]),(h_down[5],h_up[5],w_down[5],w_up[5]),(h_down[6],h_up[6],w_down[6],w_up[6]),(h_down[7],h_up[7],w_down[7],w_up[7]),(h_down[8],h_up[8],w_down[8],w_up[8])]

    return h_up,h_down,w_up,w_down,number_partition,zones

def obtaine_mac_and_val():

    g               = open(args.wifi_path + 'locations_resume.txt', 'r')

    to_find         = []
    values          = []
    been            = 0
    zone            = "no_zone"
    for line in g:
        macs   = line[0:17]
        val_2b = line[32:34]
        val_3b = line[48:50]
        if len(to_find)==0:
            to_find.append(macs)        
        else:
            for find in to_find:
                if macs==find:
                    been=1
                    if val_3b!='':
                        values.append((macs,int(val_2b),int(val_3b),zone))
            if been==0:
                to_find.append(macs)
                if val_3b!='':
                    values.append((macs,int(val_2b),int(val_3b),zone))
            else:
                been=0

    g.close()
    return to_find,values

def obtain_mac_zone():
    
    h               = open(args.zone_wifi_path + 'devices_zones.txt', 'r')

    to_find_zones   = []
    zone_number     = []
    mac_zones       = []
    values_zones    = []
    for line in h:
        mc  = line[0:17]
        zn  = line[18:20]
        to_find_zones.append((mc,zn))
        zone_number.append(0)

    for zone_mac in to_find_zones:
        vec=(zone_mac[0],int(zone_mac[1]))
        mac_zones.append(vec)
        values_zones.append(([],[]))


    h.close()
    return zone_number,mac_zones,values_zones

def determine_zones(to_find,values,mac_zones,values_zones,n_values,zone_number):
    cnt_5           = 0
    zones_file      = []
    for mac in to_find:
        zone            = "no_zone"
        values_2b_find  = []
        values_3b_find  = []
        difference      = []
        cont_2          = 0

        for comp_2 in values:
            values_2b   = []
            values_3b   = []
            cont_3      = 0

            for comp_3 in mac_zones:
                if comp_3[0]==comp_2[0]:
                    if cont_3 < n_values:
                        values_2b.append(comp_2[1])
                        values_3b.append(comp_2[2])
                    else:
                        values_2b[np.remainder(cont_3,n_values)]=comp_2[1]
                        values_3b[np.remainder(cont_3,n_values)]=comp_2[2]
                    vec=(np.mean(values_2b),np.mean(values_3b))
                    cont_3+=1
                    values_zones[comp_3[1]]=vec
                elif mac==comp_2[0]:
                    if cont_2 < n_values:
                        values_2b_find.append(comp_2[1])
                        values_3b_find.append(comp_2[2])
                    else:
                        values_2b_find[np.remainder(cont_2,n_values)]=comp_2[1]
                        values_3b_find[np.remainder(cont_2,n_values)]=comp_2[2]
                    cont_2+=1
                    vec_find = (np.mean(values_2b_find),np.mean(values_3b_find))
                    min_distance = 9999999
                    cnt_4=0
                    for cmp_val in values_zones:
                        distance=math.sqrt(np.sum(pow(abs(np.subtract(cmp_val,vec_find)),2)))  
                        if distance<min_distance:
                            min_distance=distance
                            zone=cnt_4
                        cnt_4+=1    
                    if cnt_5==2:
                        zones_file.append((mac,zone))
                        zone_number[zone]+=1
                        cnt_5=0
                    else:
                        cnt_5+=1
    return zone_number,zones_file

def wifi_data(args):

    f_auto          = open('out_automatic.txt','w')

    to_find         = [] 
    values          = []    
    zone_number     = []
    values_zones    = []
    mac_zones       = []
    been            = 0
    n_values        = 10    
    zone            = "no_zone"

    to_find,values                      = obtaine_mac_and_val()
    zone_number,mac_zones,values_zones  = obtain_mac_zone()
    zone_number,zones_file              = determine_zones(to_find,values,mac_zones,values_zones,n_values,zone_number)

    for (mac,zona) in zones_file:
        f_auto.write(str(mac)+': '+str(zona)+'\n')

    f_auto.close()
    return zone_number  

def preprocessing(h_down,h_up,w_down,w_up,gamma_vector,img,contrast_vector,args,height_min,height_fin,width):

    for ii in range(0,len(h_up)):
        img_1   = img[int(h_down[ii]):int(h_up[ii]),int(w_down[ii]):int(w_up[ii])]
        img_1   = adjust_gamma(img_1, gamma=gamma_vector[ii]) # input 
        lab     = cv2.cvtColor(img_1, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=contrast_vector[ii], tileGridSize=(args.grid_size,args.grid_size))
        cl      = clahe.apply(l)
        limg    = cv2.merge((cl,a,b))

        img[int(h_down[ii]):int(h_up[ii]),int(w_down[ii]):int(w_up[ii])] = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    img = img[int(height_min):int(height_fin),0:int(width)]
    return img

def zone_calculation(zones,t_x_bar,t_y_bar,number_partition,height_min,relation_vector):
    for ii in range(0,len(zones)):
        (h_down_zones,h_up_zones,w_down_zones,w_up_zones) = zones[ii]
        h_down_zones=int((h_down_zones - height_min))
        h_up_zones=int((h_up_zones -height_min))
        w_down_zones=int(w_down_zones)
        w_up_zones=int(w_up_zones)
        if ((w_down_zones<=t_x_bar<=w_up_zones) and (h_down_zones<=t_y_bar<=h_up_zones)):
            number_partition[ii]+=1

    zones_computer=[0,0,0]
    for jj in range(0,len(number_partition)):
        zones_computer[relation_vector[jj]-1]+=number_partition[jj]
    return zones_computer,number_partition

parser = argparse.ArgumentParser()
parser.add_argument('--cam_or_file'     , type=int,     default=1)
parser.add_argument('--grid_size'       , type=int,     default=1)
parser.add_argument('--seconds_movement', type=int,     default=5)
parser.add_argument('--detection_zone'  , type=int,     default=5) #5 for face 11 for upper body >17 for whole body 
parser.add_argument('--track_quality'   , type=int,     default=8)
parser.add_argument('--n_divisions'     , type=int,     default=6)
parser.add_argument('--reset_movement'  , type=int,     default=0)
parser.add_argument('--batch_length'    , type=int,     default=1)
parser.add_argument('--n_frames'        , type=int,     default=19)
parser.add_argument('--new_data'        , type=int,     default=30)
parser.add_argument('--model'           , type=int,     default=101)
parser.add_argument('--prueba'          , type=int,     default=300)
parser.add_argument('--threshold_zones' , type=float,   default=0.9)
parser.add_argument('--scale_factor'    , type=float,   default=0.7125)
parser.add_argument('--gamma'           , default="[2.75,2.75,2.75,2.75,2.75,2.75]")
parser.add_argument('--contrast'        , default="[15,15,15,15,15,15]")
parser.add_argument('--relation_zones'  , default="[1,1,2,2,3,3]")
parser.add_argument('--cam_id'          , default="rtsp://admin:admin1234@192.168.15.220:554/Streaming/channels/401")
parser.add_argument('--input_file_name' , default="a")
parser.add_argument('--output_file_name', default="data/track_video/track.avi")
parser.add_argument('--wifi_path'       , default="wifi_data/")
parser.add_argument('--zone_wifi_path'  , default="")
args = parser.parse_args()

def main():

    relation_vector = np.array(json.loads(args.relation_zones))
    contrast_vector = np.array(json.loads(args.contrast))
    gamma_vector    = np.array(json.loads(args.gamma))

    arguments_com(args)

    with tf.Session() as sess:

        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride            = model_cfg['output_stride']

        if args.cam_or_file == 0:
            cap = cv2.VideoCapture(args.input_file_name)
        elif args.cam_or_file==1:
            cap = cv2.VideoCapture(args.cam_id)

        fourcc              = cv2.VideoWriter_fourcc(*'DIVX')
        height              = cap.get(4)
        width               = cap.get(3)
        
        listofcenters       = []
        centers             = []
        max_displacement    = []
        max_movement_single = []
        faceTrackers        = {}
        frame_number        = 0
        currentFaceID       = 0
        cont                = 0
        width_min           = 0
        min_fid             = 0
        zones_computer      = [0,0,0]
        width_int           = width/3
        rectangleColor      = (0,0,255)
        height_fin          = height*(8/10)
        height_min          = height*(2/7)
        

        h_up,h_down,w_up,w_down,number_partition,zones = division_funct(height,width,args)

        out                 = cv2.VideoWriter(args.output_file_name, fourcc, 20.0, (int(width),int(math.ceil(height_fin-height_min))))
        len_video           = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start               = time.time()
        zone_number         = []
        cnt_6               = 0
        frame_count         = 0

        while True:
            if (int(time.time()-start)> (args.prueba)) and (args.prueba!=0):
                print('Average FPS: ', frame_count / (time.time() - start))
                print('time: ', (time.time() - start))
                exit()
            if int(time.time()-start)> ((cont+1)*args.seconds_movement):
                zone_number = wifi_data(args)

            res, img = cap.read()
            frame_count     += 1
            if not res:
                break

            img = preprocessing(h_down,h_up,w_down,w_up,gamma_vector,img,contrast_vector,args,height_min,height_fin,width)
            input_image, display_image, output_scale = posenet.process_input(img, scale_factor=args.scale_factor, output_stride=output_stride)

            cnt              = 0
            fidsToDelete     = []
            number_partition = zeros_number_partition(args)

            for fid in faceTrackers.keys():
                cnt+=1
                trackingQuality = faceTrackers[fid].update(overlay_image)
                if trackingQuality < args.track_quality:
                    if args.reset_movement==1:
                        del max_movement_single[cnt]
                        del max_displacement[cnt]
                    fidsToDelete.append(fid)

            for fid in fidsToDelete:
                faceTrackers.pop(fid,None)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(model_outputs,feed_dict={'image:0': input_image})
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses( heatmaps_result.squeeze(axis=0), offsets_result.squeeze(axis=0), displacement_fwd_result.squeeze(axis=0), displacement_bwd_result.squeeze(axis=0), output_stride=output_stride, max_pose_detections=10, min_pose_score=0.15)
            keypoint_coords *= output_scale

            overlay_image = posenet.draw_skel_and_kp(display_image, pose_scores, keypoint_scores, keypoint_coords, min_pose_score=0.15, min_part_score=0.1)                
            
            if (frame_count % args.n_frames) == 0:
                for idx,esq in enumerate(keypoint_coords):
                    x_coords = []
                    y_coords = []
                    for ii in range(0,min(args.detection_zone,len(esq)-1)):
                        if esq[ii][0]!=0 or esq[ii][1]!=0:
                            x_coords.append(esq[ii][0])
                            y_coords.append(esq[ii][1]) 
                    
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
                        matchedFid = None
                    
                        for fid in faceTrackers.keys():
                            tracked_position = faceTrackers[fid].get_position()
                            t_x= int(tracked_position.left())
                            t_y= int(tracked_position.top())
                            t_w= int(tracked_position.width())
                            t_h= int(tracked_position.height())
                            t_x_bar= t_x + 0.5 * t_w
                            t_y_bar= t_y + 0.5 * t_h
                            if ((t_y<=x_bar<=(t_y+t_h)) and (t_x<=y_bar<=(t_x+t_w)) and (x<=t_y_bar<=(x+w)) and (y<=t_x_bar<=(y+h))):
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
                            tracker = dlib.correlation_tracker()
                            tracker.start_track(overlay_image,dlib.rectangle(int(y_min), int(x_min), int(y_max) , int(x_max)))
                            faceTrackers[ currentFaceID ] = tracker
                            currentFaceID += 1
                            centers=[]
                            centers=[(int(y_bar),int(x_bar))]+centers
                            listofcenters.append(centers)
                            max_displacement.append(0)
                            max_movement_single.append(0)

                cnt_6+=1
                if cnt_6 == args.new_data:
                    for fid in faceTrackers.keys():
                        fidsToDelete.append(fid)
                    for fid in fidsToDelete:
                        faceTrackers.pop(fid,None)
                        min_fid=fid
                    cnt_6 = 0

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

                for (x,y) in centers:

                    distance=abs((pow(x_bar,2)+pow(y_bar,2))-(pow(t_x_bar,2)+pow(t_y_bar,2)))
                    
                    max_displacement[fid]+=distance
                    if distance > max_movement_single[fid]:
                        max_movement_single[fid]=distance

                zones_computer,number_partition = zone_calculation(zones,t_x_bar,t_y_bar,number_partition,height_min,relation_vector)

            if zone_number!=[]:
                for ll in range(0,len(zones_computer)):
                    if (zones_computer[ll]<(args.threshold_zones*zone_number[ll])):
                        #contrast_vector[ll] +=1
                        #gamma_vector[ll]    +=0.25
                        a=1

if __name__ == "__main__":
    main()