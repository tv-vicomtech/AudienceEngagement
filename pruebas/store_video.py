import cvlib as cv
import time
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cam'		, default="rtsp://admin:admin1234@192.168.15.220:554/Streaming/channels/402")
parser.add_argument('--output'	, default="Video.avi")
parser.add_argument('--max_time', type=float, default=300)
args = parser.parse_args()

start=time.time()
cap 				= cv2.VideoCapture(args.cam)
fourcc              = cv2.VideoWriter_fourcc(*'DIVX')
height              = cap.get(4)
width               = cap.get(3)
out                 = cv2.VideoWriter(args.output, fourcc, 10.0, (int(width),int(height)))
frame_count         = 0

while True:
	#print(time.time()-start,args.max_time)
	if (time.time()-start)>=args.max_time:
		print(time.time()-start,frame_count/(time.time()-start))
		exit()

	res, img = cap.read()
	frame_count     += 1
	out.write(img)
	# cv2.imshow("asdf",img)

	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	print(time.time()-start,frame_count/(time.time()-start))
	# 	break