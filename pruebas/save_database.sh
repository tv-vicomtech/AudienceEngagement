
count=0
count_prueba=0

#python store_video.py --cam rtsp://admin:admin1234@192.168.15.220:554/Streaming/channels/402 --output /home/usuario/code/miguel/find3/pruebas/Run_$count_prueba/Video_2_m.avi --max_time 300 &
ffmpeg -t 90 -rtsp_transport tcp -i rtsp://admin:admin1234@192.168.15.220:554/Streaming/channels/402 -vcodec copy -movflags +faststart /home/usuario/code/miguel/find3/pruebas/Run_$count_prueba/Video_2_m.mp4 -y &
#python store_video.py --cam rtsp://admin:admin1234@192.168.15.220:554/Streaming/channels/401 --output /home/usuario/code/miguel/find3/pruebas/Run_$count_prueba/Video_2_b.avi --max_time 300 &
ffmpeg -t 90 -rtsp_transport tcp -i rtsp://admin:admin1234@192.168.15.220:554/Streaming/channels/401 -vcodec copy -movflags +faststart /home/usuario/code/miguel/find3/pruebas/Run_$count_prueba/Video_2_b.mp4 -y &
while :
do
	tmp_1=$(http --timeout=300 GET 192.168.35.101:8005/api/v1/locations/run_$count_prueba)
	echo "locations called"
	echo $tmp_1 >> /home/usuario/code/miguel/find3/pruebas/Run_$count_prueba/locations_$count.json
	echo "locations stored"
	sleep 2
	tmp_2=$(http --timeout=300 GET 192.168.35.101:8005/api/v1/devices/Run_$count_prueba)
	echo "devices called"
	echo $tmp_2 >> /home/usuario/code/miguel/find3/pruebas/Run_$count_prueba/devices_$count.json
	echo "devices stored"
	sleep 2
	tmp_3=$(http --timeout=300 GET 192.168.35.101:8005/api/v1/database/Run_$count_prueba)
	echo "database called"
	echo $tmp_3 >> /home/usuario/code/miguel/find3/pruebas/Run_$count_prueba/database_$count.sql
	echo "database stored"
	sleep 2
	python process_2.py /home/usuario/code/miguel/find3/pruebas/Run_$count_prueba $count
	echo "reguex done"
	count=$((count+1))
        echo $count
        sleep 40
done
