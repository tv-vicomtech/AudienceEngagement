import sys
import re
from string import *
import time

path=sys.argv[1]
count=sys.argv[2]

start=time.time()
f   = open(path+'/locations_'+count+'.json', 'r')
g   = open(path+'/locations_simple_'+count+'.txt', 'a')
g_2 = open(path+'/locations_resume_'+count+'.txt', 'a')
values   = []
access   = []
cnt=0

for line in f:
	macs=re.findall(r'"d":"wifi-([^"]*)',line)
	devices=re.findall(r'"wifi":{"([^}]+)',line)
	for ii in range(0,(len(macs)-1)):
		a= macs[ii][0:2] + macs[ii][3:5]+ macs[ii][6:8]
		values_single = []
		access_single = []
		g.write(macs[ii] + " : " + devices[ii] + "\n")
		g_2.write(macs[ii] + " : " + devices[ii][0:11] + " " + devices[ii][19:21]+ "  " + devices[ii][23:34] + " " + devices[ii][42:44] + "\n")
		values_single.append(int(devices[ii][19:21]))
		access_single.append(devices[ii][0:11])
		if (len(devices[ii])>25):
			values_single.append(int(devices[ii][42:44]))
			access_single.append(devices[ii][23:34])
		values.append(values_single)
		access.append(access_single)
print(time.time()-start)