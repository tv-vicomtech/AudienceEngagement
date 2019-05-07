import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import argparse
from PIL import Image, ImageDraw
import time
from imageai.Detection import ObjectDetection
from ctypes import *
from PIL import Image, ImageDraw
import glob
import math
import cvlib as cv
import cv2
import random
import os 

f=open("benchmarks/obj.txt","r")

image_list = []

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("model_data/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image.encode('utf-8'), 0, 0)
    
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)

    return res, res


if __name__ == "__main__":

    f2=open("benchmarks/obj_ave.txt","w")
    filenames = [imag for imag in glob.glob("data/obj_personas/*")]

    filenames.sort() # ADD THIS LINE

    images = []
    
    net = load_net("model_data/yolov3.cfg".encode('utf-8'), "model_data/yolov3.weights".encode('utf-8'), 0)
    meta = load_meta("model_data/coco.data".encode('utf-8'))
    n=0

    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "model_data/resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    custom_objects = detector.CustomObjects(person=True)

    detector2 = ObjectDetection()
    detector2.setModelTypeAsRetinaNet()
    detector2.setModelPath( os.path.join(execution_path , "model_data/resnet50_coco_best_v2.0.1.h5"))
    detector2.loadModel(detection_speed="fast")
    custom_objects = detector2.CustomObjects(person=True)




    for imag in filenames:

        print('\n\nImagen {}\n'.format(n))
        a=int(f.readline())
        f2.write("\n\nImagen {} with {} people\n".format(n,a))

        # read input image
        image = cv2.imread(imag)

        # apply object detection

        start = time.time()
        bbox, label, conf = cv.detect_common_objects(image)
        elapsed_time = time.time() - start

        b=len(label)
        if b>a:
            acc=100*(a/b)
        else:
            acc=100*(b/a)

        print("Time for the object detection with cvlib: {}, with {} people".format(elapsed_time, len(label)))
        print("The image had {} people, so it has an accuracy of {}%".format(a, acc))
        f2.write("Time {}, people {}, accuracy {}\n".format(elapsed_time,len(label),acc))

        for i,label in enumerate(label):
    
            # Draw a box around 
            cv2.rectangle(image, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), (0,255,0), 2)
    
        cv2.imwrite('data/images_obj_det/image{}_cvlib.jpg'.format(n),image)

        image = cv2.imread(imag)
        start = time.time()
        r,d = detect(net, meta,imag)
        elapsed_time = time.time() - start

        b=len(d)
        if b>a:
            acc=100*(a/b)
        else:
            acc=100*(b/a)

        print("Time for the object detection with darknet: {}, with {} people".format(elapsed_time, len(d)))
        print("The image had {} people, so it has an accuracy of {}%".format(a, acc))
        f2.write("Time {}, people {}, accuracy {}\n".format(elapsed_time,len(label),acc))

        

        for l,d in enumerate(d):
        
            (startX,startY) = int(r[l][2][0]-r[l][2][2]/2),int(r[l][2][1]-r[l][2][3]/2)
            (endX,endY) = int(r[l][2][0]+r[l][2][2]/2),int(r[l][2][1]+r[l][2][3]/2)
            # draw rectangle over
            cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

        cv2.imwrite('data/images_obj_det/image{}_darknet.jpg'.format(n),image)

        l=0

        start=time.time()
        detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=imag, output_image_path='data/images_obj_det/image{}_image_ai_normal.jpg'.format(n), minimum_percentage_probability=45)
        elapsed_time=time.time()-start

        b=len(detections)
        if b>a:
            acc=100*(a/b)
        else:
            acc=100*(b/a)

        print("Time for the object detection with imageai normal speed: {}, with {} people".format(elapsed_time, len(detections)))
        print("The image had {} people, so it has an accuracy of {}%".format(a, acc))
        f2.write("Time {}, people {}, accuracy {}\n".format(elapsed_time,len(label),acc))


        start=time.time()
        detections = detector2.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=imag, output_image_path='data/images_obj_det/image{}_image_ai_fast.jpg'.format(n), minimum_percentage_probability=45)
        elapsed_time=time.time()-start

        b=len(detections)
        if b>a:
            acc=100*(a/b)
        else:
            acc=100*(b/a)

        print("Time for the object detection with imageai fast speed: {}, with {} people".format(elapsed_time, len(detections)))
        print("The image had {} people, so it has an accuracy of {}%".format(a, acc))
        f2.write("Time {}, people {}, accuracy {}\n".format(elapsed_time,len(label),acc))

        n+=1

