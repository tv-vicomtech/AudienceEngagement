import re
from string import *
f   = open('p1_locations.txt', 'r')
f_2 = open('mac_vendors.txt', 'r')
g   = open('p1_locations_process_fin.txt', 'w+')
g_2 = open('p1_locations_process_2_fin.txt', 'w+')
values   = []
access   = []
mac_list = []
cnt=0
for line in f_2:
	x = line.split()
	mac_list.append(x[0])
print(mac_list)
for line in f:
	macs=re.findall(r'"d":"wifi-([^"]*)',line)
	devices=re.findall(r'"wifi":{"([^}]+)',line)
	for ii in range(0,(len(macs)-1)):
		a= macs[ii][0:2] + macs[ii][3:5]+ macs[ii][6:8]
		for mac_good in mac_list :
			if a == mac_good:
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