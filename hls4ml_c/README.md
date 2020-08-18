# hls4ml on AWS FPGA Developer AMI v1.8.x (HLS C/C++ Kernel)

Setup tools, licenses, check connection to FPGA card

Check out packages
```bash
# check out hls4ml_c SDAccel project
git clone https://github.com/drankincms/hls4ml_c -b hcal_model_aws
```
Compile SDAccel project
```bash
cd hls4ml_c
make check TARGETS=sw_emu DEVICES=$AWS_PLATFORM all # software emulation
make check TARGETS=hw_emu DEVICES=$AWS_PLATFORM all # hardware emulation
make TARGETS=hw DEVICES=$AWS_PLATFORM all # build
```

Run project
```bash
./host 
```
