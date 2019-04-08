import re
from string import *
f   = open('p2_locations.txt', 'r')
g   = open('p2_locations_process.txt', 'w+')
g_2 = open('p2_locations_process_2.txt', 'w+')
values = []
access = []
for line in f:
	macs=re.findall(r'"d":"wifi-([^"]*)',line)
	devices=re.findall(r'"wifi":{"([^}]+)',line)
	print(type(macs))
	for ii in range(0,(len(macs)-1)):
		values_single = []
		access_single = []
		g.write(macs[ii] + " : " + devices[ii]+ "\n")
		g_2.write(macs[ii] + " : " + devices[ii][0:11] + " " + devices[ii][19:21]+ "  " + devices[ii][23:34] + " " + devices[ii][42:44] + "\n")
		values_single.append(atoi(devices[ii][19:21]))
		access_single.append(devices[ii][0:11])
		if (len(devices[ii])>25):
			values_single.append(atoi(devices[ii][42:44]))
			access_single.append(devices[ii][23:34])
		values.append(values_single)
		access.append(access_single)