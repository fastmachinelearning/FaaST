# FaaST Server for FACILE

Setup Instructions:

```
git clone https://github.com/grpc/grpc
cd grpc
git checkout -b v1.27.0
git submodule update --init
mkdir -p cmake/build
cd cmake/build
cmake ../..
make -j 8
cd ../../examples/
git clone https://github.com/hls-fpga-machine-learning/FaaST.git
cd FaaST
make
./server hls4ml_c/build_dir.hw.xilinx_u250_xdma_201830_2/alveo_hls4ml.xclbin
```

Modifications to the FPGA kernel can be made in `hls4ml_c` with standard Vitis Accel commands
