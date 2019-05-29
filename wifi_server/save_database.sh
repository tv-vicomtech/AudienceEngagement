count=0
while :
do
	tmp_1=$(http --timeout=300 GET localhost:8005/api/v1/locations/prueba_9_3_b)
	echo "locations called"
	echo $tmp_1 >> ~/find3/computer_vision/wifi_data/locations.json
	echo "locations stored"
	sleep 2
	tmp_2=$(http --timeout=300 GET localhost:8005/api/v1/devices/prueba_9_3_b)
	echo "devices called"
	echo $tmp_2 >> ~/find3/computer_vision/wifi_data/devices.json
	echo "devices stored"
	sleep 2
	tmp_3=$(http --timeout=300 GET localhost:8005/api/v1/database/prueba_9_3_b)
	echo "database called"
	echo $tmp_3 >> ~/find3/computer_vision/wifi_data/database.sql
	echo "database stored"
	count=$((count+1))
        echo $count
        sleep 40
done
