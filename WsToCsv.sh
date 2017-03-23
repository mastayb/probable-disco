#!/bin/bash  

filter="frame[54:6] == 07:6b:00:01:00:01 && dis.pdu_type == 1"

columns="-e frame.number -e dis.timestamp \
   -e dis.entity_linear_velocity.x -e dis.entity_linear_velocity.y \
   -e dis.entity_linear_velocity.z -e dis.entity_location.x \
   -e dis.entity_location.y -e dis.entity_location.z"

printoptions="-E header=y"


for f in $@ 
do
   if [ ${f: -5} != ".pcap"]
   then 
      continue
   fi
   of=${f/pcap/csv}
   tshark -r "$f" -Y "$filter" -T fields $columns $printoptions > $of 
done
