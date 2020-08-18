#!/bin/sh

binname=`ls build_dir*/ | grep "\.xclbin$" | sed 's/\.xclbin//'`
dirname=`ls . | grep "build_dir"`
echo $dirname
echo $binname
$VITIS_DIR/tools/create_vitis_afi.sh -xclbin=$dirname/$binname.xclbin -o=$binname -s3_bucket=fnal-afi -s3_dcp_key=inputfiles -s3_logs_key=logs 
