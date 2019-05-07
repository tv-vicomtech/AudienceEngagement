import numpy as np
import cv2
from PIL import Image

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

gamma=2.0
contrast=2.5
grid_size=1
color=0
img = cv2.imread('im_prueba.png',color)
if color==1:
	height, width,channels = img.shape
else:
	height, width = img.shape

#separation limit definition
windowed = int(max(height,width)/75)
height_fin=(height*(8/10))
width_int=width/3
height_min=height*(2/7)
width_min=0
height_int= (height_min + height_fin)/2
h_up=[(height_int + windowed),(height_int + windowed),(height_int + windowed),height_fin,height_fin,height_fin]
h_down=[height_min,height_min,height_min,(height_int - windowed),(height_int - windowed),(height_int - windowed)]
w_up=[(width_int + windowed),(width_int*2 + windowed),(width_int*3),(width_int + windowed),(width_int*2 + windowed),(width_int*3)]
w_down=[width_min,(width_int - windowed),(width_int*2 - windowed),width_min,(width_int - windowed),(width_int*2 - windowed)]
#image separation
#img_1 = img[int(height_min):int(height_int + windowed),int(width_min):int(width_int + windowed)]
#img_2 = img[int(height_min):int(height_int + windowed),int(width_int - windowed):int(width_int*2 + windowed)]
#img_3 = img[int(height_min):int(height_int + windowed),int(width_int*2 - windowed):int(width_int*3)]
#img_4 = img[int(height_int - windowed):int(height_fin),int(width_min):int(width_int + windowed)]
#img_5 = img[int(height_int - windowed):int(height_fin),int(width_int - windowed):int(width_int*2 + windowed)]
#img_6 = img[int(height_int - windowed):int(height_fin),int(width_int*2 - windowed):int(width_int*3)]

#brightness calculation
#brightness_1=img_1.mean()
#brightness_2=img_2.mean()
#brightness_3=img_3.mean()
#brightness_4=img_4.mean()
#brightness_5=img_5.mean()
#brightness_6=img_6.mean()

for ii in range(0,len(h_up)-1):
	img_1=img[int(h_down[ii]):int(h_up[ii]),int(w_down[ii]):int(w_up[ii])]
	#gamma adjust

	Gamma_adjusted = adjust_gamma(img_1, gamma=gamma)
	cv2.imshow("Gamma", np.hstack([img_1, Gamma_adjusted]))

	#Contrast maximization

	if color==1:
		lab= cv2.cvtColor(Gamma_adjusted, cv2.COLOR_BGR2LAB)
		l, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(grid_size,grid_size))
		cl = clahe.apply(l)
		limg = cv2.merge((cl,a,b))
		final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	else:
		clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(grid_size,grid_size))
		final = clahe.apply(Gamma_adjusted)

	cv2.imshow("Contrast"+str(ii), np.hstack([img_1, final]))
	

cv2.waitKey(0)
cv2.destroyAllWindows()

