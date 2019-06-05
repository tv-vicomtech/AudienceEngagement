BEGIN TRANSACTION; CREATE TABLE devices (id TEXT PRIMARY KEY, name TEXT); INSERT INTO "devices" VALUES('b','wifi-da:a1:19:9d:aa:95'); INSERT INTO "devices" VALUES('c','wifi-da:a1:19:84:51:6b'); INSERT INTO "devices" VALUES('d','wifi-da:a1:19:a2:66:25'); INSERT INTO "devices" VALUES('e','wifi-da:a1:19:ef:4f:01'); INSERT INTO "devices" VALUES('f','wifi-1c:87:2c:67:b5:98'); INSERT INTO "devices" VALUES('g','wifi-da:a1:19:80:07:7b'); INSERT INTO "devices" VALUES('h','wifi-da:a1:19:3a:0f:8a'); INSERT INTO "devices" VALUES('i','wifi-da:a1:19:36:72:87'); INSERT INTO "devices" VALUES('j','wifi-da:a1:19:bc:33:f2'); INSERT INTO "devices" VALUES('k','wifi-da:a1:19:f3:b9:8e'); INSERT INTO "devices" VALUES('l','wifi-da:a1:19:29:70:55'); INSERT INTO "devices" VALUES('m','wifi-d8:5b:2a:47:25:bb'); INSERT INTO "devices" VALUES('n','wifi-d0:59:e4:4e:24:ca'); INSERT INTO "devices" VALUES('o','wifi-64:a2:f9:6a:13:be'); INSERT INTO "devices" VALUES('p','wifi-da:a1:19:b1:37:8e'); INSERT INTO "devices" VALUES('q','wifi-da:a1:19:bd:ee:14'); INSERT INTO "devices" VALUES('r','wifi-48:60:5f:7d:97:63'); INSERT INTO "devices" VALUES('s','wifi-f4:7d:ef:6f:db:a1'); INSERT INTO "devices" VALUES('t','wifi-da:a1:19:4f:73:32'); INSERT INTO "devices" VALUES('u','wifi-da:a1:19:2c:7d:1e'); INSERT INTO "devices" VALUES('v','wifi-da:a1:19:81:c4:2b'); INSERT INTO "devices" VALUES('w','wifi-da:a1:19:2e:72:50'); INSERT INTO "devices" VALUES('x','wifi-da:a1:19:a5:7b:c6'); INSERT INTO "devices" VALUES('y','wifi-da:a1:19:eb:c6:94'); INSERT INTO "devices" VALUES('z','wifi-40:88:05:bf:18:0c'); INSERT INTO "devices" VALUES('A','wifi-da:a1:19:da:33:89'); INSERT INTO "devices" VALUES('B','wifi-da:a1:19:d8:de:1a'); INSERT INTO "devices" VALUES('C','wifi-da:a1:19:d2:82:d3'); INSERT INTO "devices" VALUES('D','wifi-dc:ee:06:7f:e2:37'); INSERT INTO "devices" VALUES('E','wifi-18:65:90:81:65:7b'); INSERT INTO "devices" VALUES('F','wifi-da:a1:19:1a:26:38'); INSERT INTO "devices" VALUES('G','wifi-da:a1:19:7a:f8:05'); INSERT INTO "devices" VALUES('H','wifi-18:f0:e4:e3:6a:33'); INSERT INTO "devices" VALUES('I','wifi-da:a1:19:c8:bb:fe'); INSERT INTO "devices" VALUES('J','wifi-da:a1:19:49:13:4b'); CREATE TABLE gps (id INTEGER PRIMARY KEY, timestamp INTEGER, mac TEXT, loc TEXT, lat REAL, lon REAL, alt REAL); CREATE TABLE keystore (key text not null primary key, value text); INSERT INTO "keystore" VALUES('sensorDataStringSizer','"{\"encoding\":{\"raspberry2b-wifi\":\"a\"},\"current\":1}"'); INSERT INTO "keystore" VALUES('ReverseRollingData','{"HasData":true,"Family":"run_0","Datas":[{"t":1558282311942,"f":"run_0","d":"raspberry2b","s":{"wifi":{"1c:87:2c:67:b5:98":-44,"40:88:05:bf:18:0c":-79,"da:a1:19:43:7c:a2":-85}},"gps":{}},{"t":1558282322003,"f":"run_0","d":"raspberry2b","s":{"wifi":{"1c:87:2c:67:b5:98":-43,"48:60:5f:7d:97:63":-80,"60:ab:67:f6:e4:6a":-57,"da:a1:19:d0:b6:60":-86}},"gps":{}}],"Timestamp":"2019-06-05T11:07:11.980227027Z","TimeBlock":90000000000,"MinimumPassive":0,"DeviceLocation":{},"DeviceGPS":null}'); CREATE TABLE location_predictions (timestamp integer NOT NULL PRIMARY KEY, prediction TEXT, UNIQUE(timestamp)); CREATE TABLE locations (id TEXT PRIMARY KEY, name TEXT); CREATE TABLE sensors (timestamp integer not null primary key, deviceid text, locationid text, wifi text, unique(timestamp)); INSERT INTO "sensors" VALUES(1559732821961,'u','','"a":-63'); INSERT INTO "sensors" VALUES(1559732821971,'i','','"a":-58'); INSERT INTO "sensors" VALUES(1559732821981,'H','','"a":-58'); INSERT INTO "sensors" VALUES(1559732821991,'f','','"a":-44'); INSERT INTO "sensors" VALUES(1559732822001,'z','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822011,'m','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822021,'p','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822031,'j','','"a":-54'); INSERT INTO "sensors" VALUES(1559732822042,'I','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822052,'q','','"a":-72'); INSERT INTO "sensors" VALUES(1559732822062,'A','','"a":-84'); INSERT INTO "sensors" VALUES(1559732822072,'k','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822082,'n','','"a":-88'); INSERT INTO "sensors" VALUES(1559732822092,'J','','"a":-70'); INSERT INTO "sensors" VALUES(1559732822103,'v','','"a":-82'); INSERT INTO "sensors" VALUES(1559732822113,'c','','"a":-55'); INSERT INTO "sensors" VALUES(1559732822123,'d','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822133,'C','','"a":-81'); INSERT INTO "sensors" VALUES(1559732822143,'e','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822153,'D','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822164,'r','','"a":-83'); INSERT INTO "sensors" VALUES(1559732822174,'w','','"a":-85'); INSERT INTO "sensors" VALUES(1559732822184,'g','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822194,'B','','"a":-83'); INSERT INTO "sensors" VALUES(1559732822204,'s','','"a":-75'); INSERT INTO "sensors" VALUES(1559732822214,'E','','"a":-87'); INSERT INTO "sensors" VALUES(1559732822224,'x','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822235,'h','','"a":-73'); INSERT INTO "sensors" VALUES(1559732822245,'b','','"a":-90'); INSERT INTO "sensors" VALUES(1559732822255,'F','','"a":-52'); INSERT INTO "sensors" VALUES(1559732822265,'G','','"a":-49'); INSERT INTO "sensors" VALUES(1559732822275,'o','','"a":-46'); INSERT INTO "sensors" VALUES(1559732822285,'t','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822295,'y','','"a":-65'); INSERT INTO "sensors" VALUES(1559732822306,'l','','"a":-53'); CREATE INDEX keystore_idx on keystore(key); CREATE INDEX devices_name on devices (name); CREATE INDEX sensors_devices ON sensors (deviceid); COMMIT;
BEGIN TRANSACTION; CREATE TABLE devices (id TEXT PRIMARY KEY, name TEXT); INSERT INTO "devices" VALUES('b','wifi-da:a1:19:9d:aa:95'); INSERT INTO "devices" VALUES('c','wifi-da:a1:19:84:51:6b'); INSERT INTO "devices" VALUES('d','wifi-da:a1:19:a2:66:25'); INSERT INTO "devices" VALUES('e','wifi-da:a1:19:ef:4f:01'); INSERT INTO "devices" VALUES('f','wifi-1c:87:2c:67:b5:98'); INSERT INTO "devices" VALUES('g','wifi-da:a1:19:80:07:7b'); INSERT INTO "devices" VALUES('h','wifi-da:a1:19:3a:0f:8a'); INSERT INTO "devices" VALUES('i','wifi-da:a1:19:36:72:87'); INSERT INTO "devices" VALUES('j','wifi-da:a1:19:bc:33:f2'); INSERT INTO "devices" VALUES('k','wifi-da:a1:19:f3:b9:8e'); INSERT INTO "devices" VALUES('l','wifi-da:a1:19:29:70:55'); INSERT INTO "devices" VALUES('m','wifi-d8:5b:2a:47:25:bb'); INSERT INTO "devices" VALUES('n','wifi-d0:59:e4:4e:24:ca'); INSERT INTO "devices" VALUES('o','wifi-64:a2:f9:6a:13:be'); INSERT INTO "devices" VALUES('p','wifi-da:a1:19:b1:37:8e'); INSERT INTO "devices" VALUES('q','wifi-da:a1:19:bd:ee:14'); INSERT INTO "devices" VALUES('r','wifi-48:60:5f:7d:97:63'); INSERT INTO "devices" VALUES('s','wifi-f4:7d:ef:6f:db:a1'); INSERT INTO "devices" VALUES('t','wifi-da:a1:19:4f:73:32'); INSERT INTO "devices" VALUES('u','wifi-da:a1:19:2c:7d:1e'); INSERT INTO "devices" VALUES('v','wifi-da:a1:19:81:c4:2b'); INSERT INTO "devices" VALUES('w','wifi-da:a1:19:2e:72:50'); INSERT INTO "devices" VALUES('x','wifi-da:a1:19:a5:7b:c6'); INSERT INTO "devices" VALUES('y','wifi-da:a1:19:eb:c6:94'); INSERT INTO "devices" VALUES('z','wifi-40:88:05:bf:18:0c'); INSERT INTO "devices" VALUES('A','wifi-da:a1:19:da:33:89'); INSERT INTO "devices" VALUES('B','wifi-da:a1:19:d8:de:1a'); INSERT INTO "devices" VALUES('C','wifi-da:a1:19:d2:82:d3'); INSERT INTO "devices" VALUES('D','wifi-dc:ee:06:7f:e2:37'); INSERT INTO "devices" VALUES('E','wifi-18:65:90:81:65:7b'); INSERT INTO "devices" VALUES('F','wifi-da:a1:19:1a:26:38'); INSERT INTO "devices" VALUES('G','wifi-da:a1:19:7a:f8:05'); INSERT INTO "devices" VALUES('H','wifi-18:f0:e4:e3:6a:33'); INSERT INTO "devices" VALUES('I','wifi-da:a1:19:c8:bb:fe'); INSERT INTO "devices" VALUES('J','wifi-da:a1:19:49:13:4b'); CREATE TABLE gps (id INTEGER PRIMARY KEY, timestamp INTEGER, mac TEXT, loc TEXT, lat REAL, lon REAL, alt REAL); CREATE TABLE keystore (key text not null primary key, value text); INSERT INTO "keystore" VALUES('sensorDataStringSizer','"{\"encoding\":{\"raspberry2b-wifi\":\"a\"},\"current\":1}"'); INSERT INTO "keystore" VALUES('ReverseRollingData','{"HasData":true,"Family":"run_0","Datas":[{"t":1558282311942,"f":"run_0","d":"raspberry2b","s":{"wifi":{"1c:87:2c:67:b5:98":-44,"40:88:05:bf:18:0c":-79,"da:a1:19:43:7c:a2":-85}},"gps":{}},{"t":1558282322003,"f":"run_0","d":"raspberry2b","s":{"wifi":{"1c:87:2c:67:b5:98":-43,"48:60:5f:7d:97:63":-80,"60:ab:67:f6:e4:6a":-57,"da:a1:19:d0:b6:60":-86}},"gps":{}}],"Timestamp":"2019-06-05T11:07:11.980227027Z","TimeBlock":90000000000,"MinimumPassive":0,"DeviceLocation":{},"DeviceGPS":null}'); CREATE TABLE location_predictions (timestamp integer NOT NULL PRIMARY KEY, prediction TEXT, UNIQUE(timestamp)); CREATE TABLE locations (id TEXT PRIMARY KEY, name TEXT); CREATE TABLE sensors (timestamp integer not null primary key, deviceid text, locationid text, wifi text, unique(timestamp)); INSERT INTO "sensors" VALUES(1559732821961,'u','','"a":-63'); INSERT INTO "sensors" VALUES(1559732821971,'i','','"a":-58'); INSERT INTO "sensors" VALUES(1559732821981,'H','','"a":-58'); INSERT INTO "sensors" VALUES(1559732821991,'f','','"a":-44'); INSERT INTO "sensors" VALUES(1559732822001,'z','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822011,'m','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822021,'p','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822031,'j','','"a":-54'); INSERT INTO "sensors" VALUES(1559732822042,'I','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822052,'q','','"a":-72'); INSERT INTO "sensors" VALUES(1559732822062,'A','','"a":-84'); INSERT INTO "sensors" VALUES(1559732822072,'k','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822082,'n','','"a":-88'); INSERT INTO "sensors" VALUES(1559732822092,'J','','"a":-70'); INSERT INTO "sensors" VALUES(1559732822103,'v','','"a":-82'); INSERT INTO "sensors" VALUES(1559732822113,'c','','"a":-55'); INSERT INTO "sensors" VALUES(1559732822123,'d','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822133,'C','','"a":-81'); INSERT INTO "sensors" VALUES(1559732822143,'e','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822153,'D','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822164,'r','','"a":-83'); INSERT INTO "sensors" VALUES(1559732822174,'w','','"a":-85'); INSERT INTO "sensors" VALUES(1559732822184,'g','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822194,'B','','"a":-83'); INSERT INTO "sensors" VALUES(1559732822204,'s','','"a":-75'); INSERT INTO "sensors" VALUES(1559732822214,'E','','"a":-87'); INSERT INTO "sensors" VALUES(1559732822224,'x','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822235,'h','','"a":-73'); INSERT INTO "sensors" VALUES(1559732822245,'b','','"a":-90'); INSERT INTO "sensors" VALUES(1559732822255,'F','','"a":-52'); INSERT INTO "sensors" VALUES(1559732822265,'G','','"a":-49'); INSERT INTO "sensors" VALUES(1559732822275,'o','','"a":-46'); INSERT INTO "sensors" VALUES(1559732822285,'t','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822295,'y','','"a":-65'); INSERT INTO "sensors" VALUES(1559732822306,'l','','"a":-53'); CREATE INDEX keystore_idx on keystore(key); CREATE INDEX devices_name on devices (name); CREATE INDEX sensors_devices ON sensors (deviceid); COMMIT;
BEGIN TRANSACTION; CREATE TABLE devices (id TEXT PRIMARY KEY, name TEXT); INSERT INTO "devices" VALUES('b','wifi-da:a1:19:9d:aa:95'); INSERT INTO "devices" VALUES('c','wifi-da:a1:19:84:51:6b'); INSERT INTO "devices" VALUES('d','wifi-da:a1:19:a2:66:25'); INSERT INTO "devices" VALUES('e','wifi-da:a1:19:ef:4f:01'); INSERT INTO "devices" VALUES('f','wifi-1c:87:2c:67:b5:98'); INSERT INTO "devices" VALUES('g','wifi-da:a1:19:80:07:7b'); INSERT INTO "devices" VALUES('h','wifi-da:a1:19:3a:0f:8a'); INSERT INTO "devices" VALUES('i','wifi-da:a1:19:36:72:87'); INSERT INTO "devices" VALUES('j','wifi-da:a1:19:bc:33:f2'); INSERT INTO "devices" VALUES('k','wifi-da:a1:19:f3:b9:8e'); INSERT INTO "devices" VALUES('l','wifi-da:a1:19:29:70:55'); INSERT INTO "devices" VALUES('m','wifi-d8:5b:2a:47:25:bb'); INSERT INTO "devices" VALUES('n','wifi-d0:59:e4:4e:24:ca'); INSERT INTO "devices" VALUES('o','wifi-64:a2:f9:6a:13:be'); INSERT INTO "devices" VALUES('p','wifi-da:a1:19:b1:37:8e'); INSERT INTO "devices" VALUES('q','wifi-da:a1:19:bd:ee:14'); INSERT INTO "devices" VALUES('r','wifi-48:60:5f:7d:97:63'); INSERT INTO "devices" VALUES('s','wifi-f4:7d:ef:6f:db:a1'); INSERT INTO "devices" VALUES('t','wifi-da:a1:19:4f:73:32'); INSERT INTO "devices" VALUES('u','wifi-da:a1:19:2c:7d:1e'); INSERT INTO "devices" VALUES('v','wifi-da:a1:19:81:c4:2b'); INSERT INTO "devices" VALUES('w','wifi-da:a1:19:2e:72:50'); INSERT INTO "devices" VALUES('x','wifi-da:a1:19:a5:7b:c6'); INSERT INTO "devices" VALUES('y','wifi-da:a1:19:eb:c6:94'); INSERT INTO "devices" VALUES('z','wifi-40:88:05:bf:18:0c'); INSERT INTO "devices" VALUES('A','wifi-da:a1:19:da:33:89'); INSERT INTO "devices" VALUES('B','wifi-da:a1:19:d8:de:1a'); INSERT INTO "devices" VALUES('C','wifi-da:a1:19:d2:82:d3'); INSERT INTO "devices" VALUES('D','wifi-dc:ee:06:7f:e2:37'); INSERT INTO "devices" VALUES('E','wifi-18:65:90:81:65:7b'); INSERT INTO "devices" VALUES('F','wifi-da:a1:19:1a:26:38'); INSERT INTO "devices" VALUES('G','wifi-da:a1:19:7a:f8:05'); INSERT INTO "devices" VALUES('H','wifi-18:f0:e4:e3:6a:33'); INSERT INTO "devices" VALUES('I','wifi-da:a1:19:c8:bb:fe'); INSERT INTO "devices" VALUES('J','wifi-da:a1:19:49:13:4b'); CREATE TABLE gps (id INTEGER PRIMARY KEY, timestamp INTEGER, mac TEXT, loc TEXT, lat REAL, lon REAL, alt REAL); CREATE TABLE keystore (key text not null primary key, value text); INSERT INTO "keystore" VALUES('sensorDataStringSizer','"{\"encoding\":{\"raspberry2b-wifi\":\"a\"},\"current\":1}"'); INSERT INTO "keystore" VALUES('ReverseRollingData','{"HasData":true,"Family":"run_0","Datas":[{"t":1558282311942,"f":"run_0","d":"raspberry2b","s":{"wifi":{"1c:87:2c:67:b5:98":-44,"40:88:05:bf:18:0c":-79,"da:a1:19:43:7c:a2":-85}},"gps":{}},{"t":1558282322003,"f":"run_0","d":"raspberry2b","s":{"wifi":{"1c:87:2c:67:b5:98":-43,"48:60:5f:7d:97:63":-80,"60:ab:67:f6:e4:6a":-57,"da:a1:19:d0:b6:60":-86}},"gps":{}}],"Timestamp":"2019-06-05T11:07:11.980227027Z","TimeBlock":90000000000,"MinimumPassive":0,"DeviceLocation":{},"DeviceGPS":null}'); CREATE TABLE location_predictions (timestamp integer NOT NULL PRIMARY KEY, prediction TEXT, UNIQUE(timestamp)); CREATE TABLE locations (id TEXT PRIMARY KEY, name TEXT); CREATE TABLE sensors (timestamp integer not null primary key, deviceid text, locationid text, wifi text, unique(timestamp)); INSERT INTO "sensors" VALUES(1559732821961,'u','','"a":-63'); INSERT INTO "sensors" VALUES(1559732821971,'i','','"a":-58'); INSERT INTO "sensors" VALUES(1559732821981,'H','','"a":-58'); INSERT INTO "sensors" VALUES(1559732821991,'f','','"a":-44'); INSERT INTO "sensors" VALUES(1559732822001,'z','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822011,'m','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822021,'p','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822031,'j','','"a":-54'); INSERT INTO "sensors" VALUES(1559732822042,'I','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822052,'q','','"a":-72'); INSERT INTO "sensors" VALUES(1559732822062,'A','','"a":-84'); INSERT INTO "sensors" VALUES(1559732822072,'k','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822082,'n','','"a":-88'); INSERT INTO "sensors" VALUES(1559732822092,'J','','"a":-70'); INSERT INTO "sensors" VALUES(1559732822103,'v','','"a":-82'); INSERT INTO "sensors" VALUES(1559732822113,'c','','"a":-55'); INSERT INTO "sensors" VALUES(1559732822123,'d','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822133,'C','','"a":-81'); INSERT INTO "sensors" VALUES(1559732822143,'e','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822153,'D','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822164,'r','','"a":-83'); INSERT INTO "sensors" VALUES(1559732822174,'w','','"a":-85'); INSERT INTO "sensors" VALUES(1559732822184,'g','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822194,'B','','"a":-83'); INSERT INTO "sensors" VALUES(1559732822204,'s','','"a":-75'); INSERT INTO "sensors" VALUES(1559732822214,'E','','"a":-87'); INSERT INTO "sensors" VALUES(1559732822224,'x','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822235,'h','','"a":-73'); INSERT INTO "sensors" VALUES(1559732822245,'b','','"a":-90'); INSERT INTO "sensors" VALUES(1559732822255,'F','','"a":-52'); INSERT INTO "sensors" VALUES(1559732822265,'G','','"a":-49'); INSERT INTO "sensors" VALUES(1559732822275,'o','','"a":-46'); INSERT INTO "sensors" VALUES(1559732822285,'t','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822295,'y','','"a":-65'); INSERT INTO "sensors" VALUES(1559732822306,'l','','"a":-53'); CREATE INDEX keystore_idx on keystore(key); CREATE INDEX devices_name on devices (name); CREATE INDEX sensors_devices ON sensors (deviceid); COMMIT;
BEGIN TRANSACTION; CREATE TABLE devices (id TEXT PRIMARY KEY, name TEXT); INSERT INTO "devices" VALUES('b','wifi-da:a1:19:9d:aa:95'); INSERT INTO "devices" VALUES('c','wifi-da:a1:19:84:51:6b'); INSERT INTO "devices" VALUES('d','wifi-da:a1:19:a2:66:25'); INSERT INTO "devices" VALUES('e','wifi-da:a1:19:ef:4f:01'); INSERT INTO "devices" VALUES('f','wifi-1c:87:2c:67:b5:98'); INSERT INTO "devices" VALUES('g','wifi-da:a1:19:80:07:7b'); INSERT INTO "devices" VALUES('h','wifi-da:a1:19:3a:0f:8a'); INSERT INTO "devices" VALUES('i','wifi-da:a1:19:36:72:87'); INSERT INTO "devices" VALUES('j','wifi-da:a1:19:bc:33:f2'); INSERT INTO "devices" VALUES('k','wifi-da:a1:19:f3:b9:8e'); INSERT INTO "devices" VALUES('l','wifi-da:a1:19:29:70:55'); INSERT INTO "devices" VALUES('m','wifi-d8:5b:2a:47:25:bb'); INSERT INTO "devices" VALUES('n','wifi-d0:59:e4:4e:24:ca'); INSERT INTO "devices" VALUES('o','wifi-64:a2:f9:6a:13:be'); INSERT INTO "devices" VALUES('p','wifi-da:a1:19:b1:37:8e'); INSERT INTO "devices" VALUES('q','wifi-da:a1:19:bd:ee:14'); INSERT INTO "devices" VALUES('r','wifi-48:60:5f:7d:97:63'); INSERT INTO "devices" VALUES('s','wifi-f4:7d:ef:6f:db:a1'); INSERT INTO "devices" VALUES('t','wifi-da:a1:19:4f:73:32'); INSERT INTO "devices" VALUES('u','wifi-da:a1:19:2c:7d:1e'); INSERT INTO "devices" VALUES('v','wifi-da:a1:19:81:c4:2b'); INSERT INTO "devices" VALUES('w','wifi-da:a1:19:2e:72:50'); INSERT INTO "devices" VALUES('x','wifi-da:a1:19:a5:7b:c6'); INSERT INTO "devices" VALUES('y','wifi-da:a1:19:eb:c6:94'); INSERT INTO "devices" VALUES('z','wifi-40:88:05:bf:18:0c'); INSERT INTO "devices" VALUES('A','wifi-da:a1:19:da:33:89'); INSERT INTO "devices" VALUES('B','wifi-da:a1:19:d8:de:1a'); INSERT INTO "devices" VALUES('C','wifi-da:a1:19:d2:82:d3'); INSERT INTO "devices" VALUES('D','wifi-dc:ee:06:7f:e2:37'); INSERT INTO "devices" VALUES('E','wifi-18:65:90:81:65:7b'); INSERT INTO "devices" VALUES('F','wifi-da:a1:19:1a:26:38'); INSERT INTO "devices" VALUES('G','wifi-da:a1:19:7a:f8:05'); INSERT INTO "devices" VALUES('H','wifi-18:f0:e4:e3:6a:33'); INSERT INTO "devices" VALUES('I','wifi-da:a1:19:c8:bb:fe'); INSERT INTO "devices" VALUES('J','wifi-da:a1:19:49:13:4b'); CREATE TABLE gps (id INTEGER PRIMARY KEY, timestamp INTEGER, mac TEXT, loc TEXT, lat REAL, lon REAL, alt REAL); CREATE TABLE keystore (key text not null primary key, value text); INSERT INTO "keystore" VALUES('sensorDataStringSizer','"{\"encoding\":{\"raspberry2b-wifi\":\"a\"},\"current\":1}"'); INSERT INTO "keystore" VALUES('ReverseRollingData','{"HasData":true,"Family":"run_0","Datas":[{"t":1558282311942,"f":"run_0","d":"raspberry2b","s":{"wifi":{"1c:87:2c:67:b5:98":-44,"40:88:05:bf:18:0c":-79,"da:a1:19:43:7c:a2":-85}},"gps":{}},{"t":1558282322003,"f":"run_0","d":"raspberry2b","s":{"wifi":{"1c:87:2c:67:b5:98":-43,"48:60:5f:7d:97:63":-80,"60:ab:67:f6:e4:6a":-57,"da:a1:19:d0:b6:60":-86}},"gps":{}}],"Timestamp":"2019-06-05T11:07:11.980227027Z","TimeBlock":90000000000,"MinimumPassive":0,"DeviceLocation":{},"DeviceGPS":null}'); CREATE TABLE location_predictions (timestamp integer NOT NULL PRIMARY KEY, prediction TEXT, UNIQUE(timestamp)); CREATE TABLE locations (id TEXT PRIMARY KEY, name TEXT); CREATE TABLE sensors (timestamp integer not null primary key, deviceid text, locationid text, wifi text, unique(timestamp)); INSERT INTO "sensors" VALUES(1559732821961,'u','','"a":-63'); INSERT INTO "sensors" VALUES(1559732821971,'i','','"a":-58'); INSERT INTO "sensors" VALUES(1559732821981,'H','','"a":-58'); INSERT INTO "sensors" VALUES(1559732821991,'f','','"a":-44'); INSERT INTO "sensors" VALUES(1559732822001,'z','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822011,'m','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822021,'p','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822031,'j','','"a":-54'); INSERT INTO "sensors" VALUES(1559732822042,'I','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822052,'q','','"a":-72'); INSERT INTO "sensors" VALUES(1559732822062,'A','','"a":-84'); INSERT INTO "sensors" VALUES(1559732822072,'k','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822082,'n','','"a":-88'); INSERT INTO "sensors" VALUES(1559732822092,'J','','"a":-70'); INSERT INTO "sensors" VALUES(1559732822103,'v','','"a":-82'); INSERT INTO "sensors" VALUES(1559732822113,'c','','"a":-55'); INSERT INTO "sensors" VALUES(1559732822123,'d','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822133,'C','','"a":-81'); INSERT INTO "sensors" VALUES(1559732822143,'e','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822153,'D','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822164,'r','','"a":-83'); INSERT INTO "sensors" VALUES(1559732822174,'w','','"a":-85'); INSERT INTO "sensors" VALUES(1559732822184,'g','','"a":-78'); INSERT INTO "sensors" VALUES(1559732822194,'B','','"a":-83'); INSERT INTO "sensors" VALUES(1559732822204,'s','','"a":-75'); INSERT INTO "sensors" VALUES(1559732822214,'E','','"a":-87'); INSERT INTO "sensors" VALUES(1559732822224,'x','','"a":-79'); INSERT INTO "sensors" VALUES(1559732822235,'h','','"a":-73'); INSERT INTO "sensors" VALUES(1559732822245,'b','','"a":-90'); INSERT INTO "sensors" VALUES(1559732822255,'F','','"a":-52'); INSERT INTO "sensors" VALUES(1559732822265,'G','','"a":-49'); INSERT INTO "sensors" VALUES(1559732822275,'o','','"a":-46'); INSERT INTO "sensors" VALUES(1559732822285,'t','','"a":-86'); INSERT INTO "sensors" VALUES(1559732822295,'y','','"a":-65'); INSERT INTO "sensors" VALUES(1559732822306,'l','','"a":-53'); CREATE INDEX keystore_idx on keystore(key); CREATE INDEX devices_name on devices (name); CREATE INDEX sensors_devices ON sensors (deviceid); COMMIT;
