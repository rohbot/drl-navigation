import os
cmd = "mosquitto_pub -h rohbot.cc -p 1341 -u rohbot -P mqttbr0ker -t drl/hb -m "

os.system(cmd + " hello")