import time
start=time.time()
f = open('mac_vendors.txt', 'r')
count=0;
for line in f:
	x=line.split(" ")
	if (x[1]=="ACER TECHNOLOGIES CORP."):
		count+=1
f.close()
print("tiempo: ", time.time()-start)