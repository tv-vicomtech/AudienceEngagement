import sys
import re
from string import *
import time
import math
import numpy as np

path=sys.argv[1]



f = open('devices.txt', 'r')
g = open(path+'/llocations_resume.txt', 'r')
h = open('devices_zones.txt', 'r')

to_find  		= [] 
to_find_zones  	= [] 
values 	 		= []
zone 			= "no_zone"
been 			= 0
for line in g:
	macs=line[0:17]
	val_2b=line[32:34]
	val_3b=line[48:50]
	if len(to_find)==0:
		to_find.append(macs)
	else:
		for find in to_find:
			if macs==find:
				been=1
				if val_3b=='':
					val_3b='0'
				values.append((macs,atoi(val_2b),atoi(val_3b),zone))
		if been==0:
			to_find.append(macs)
			if val_3b=='':
				val_3b='0'
			values.append((macs,atoi(val_2b),atoi(val_3b),zone))
		else:
			been=0

for line in h:
	mc=line[0:17]
	zn=line[18:20]
	to_find_zones.append((mc,zn))

## Manual mode
print("Manual mode")
start=time.time()
cont=0
for comp in values:
	if (comp[1]<=50 and comp[2]>=60):
		zone=1
	elif (comp[1]<=55 and comp[2]>=50):
		zone=2
	else:# (comp[1]<=60 and comp[2]<=50):
		zone=3
	change=values[cont]
	change_2=(change[0],change[1],change[2],zone)
	values[cont]=change_2
	cont+=1
print(values)
print("time:",time.time()-start)
## Manual mode with mean of last n_values values

print("Manual mode with mean of last 10 values")
start=time.time()
n_values=10
zones = []
for mac in to_find:
	values_2b	= []
	values_3b	= []
	cont_2		= 0
	for comp_2 in values:
		if mac==comp_2[0]:
			if cont_2 < n_values:
				values_2b.append(comp_2[1])
				values_3b.append(comp_2[2])
			else:
				values_2b[np.remainder(cont_2,n_values)]=comp_2[1]
				values_3b[np.remainder(cont_2,n_values)]=comp_2[2]
			v_2b=np.mean(values_2b)
			v_3b=np.mean(values_3b)

			if (v_2b<=50 and v_3b>=60):
				zone=1
			elif (v_2b<=55 and v_3b>=50):
				zone=2
			else :#(v_2b<=60 and v_3b<=50):
				zone=3
			zones.append((mac,zone))
			cont_2+=1
print(zones)
print("time:",time.time()-start)

## Automatic with the devices in devices_zones.txt as reference

print("Automatic with the devices in devices_zones.txt as reference")
start=time.time()
n_values 		= 10
zones 			= []
values_zones 	= []
mac_zones 		= []
cnt_5 			= 0

for zone_mac in to_find_zones:
	vec=(zone_mac[0],atoi(zone_mac[1]))
	mac_zones.append(vec)
	values_zones.append(([],[]))

for mac in to_find:
	values_2b_find	= []
	values_3b_find	= []
	difference 		= []
	cont_2			= 0
	for comp_2 in values:
		values_2b	= []
		values_3b	= []
		cont_3		= 0
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
				values_zones[comp_3[1]-1]=vec
			if mac==comp_2[0]:
				if cont_2 < n_values:
					values_2b_find.append(comp_2[1])
					values_3b_find.append(comp_2[2])
				else:
					values_2b_find[np.remainder(cont_2,n_values)]=comp_2[1]
					values_3b_find[np.remainder(cont_2,n_values)]=comp_2[2]
				cont_2+=1
				vec_find = (np.mean(values_2b),np.mean(values_3b))
				min_distance = 9999999
				cnt_4=0
				for cmp_val in values_zones:
					cnt_4+=1
					if(cmp_val!=([],[])):
						distance=math.sqrt(np.sum(pow(abs(np.subtract(cmp_val,vec_find)),2)))  
						if distance<min_distance:
							min_distance=distance
							zone=cnt_4
				if cnt_5==2:
					zones.append((mac,zone))
					cnt_5=0
				else:
					cnt_5+=1
print(zones)
print("time:",time.time()-start)