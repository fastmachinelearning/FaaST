#!/bin/sh

afiid=`cat *_afi_id.txt | grep -m 1 "FpgaImageId" | tail -n 1 | sed 's/^.*"\(afi-.*\)",/\1/'`
echo $afiid
aws ec2 describe-fpga-images --region us-west-2 --fpga-image-ids $afiid
