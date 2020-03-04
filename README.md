git clone https://github.com/grpc/grpc

cd grpc

git checkout -b v1.27.0

git submodule update --init

mkdir -p cmake/build

cd cmake/build

cmake ../..

make -j 8

cd ../../examples/

git clone https://github.com/drankincms/grpc-trt-fgpa.git

cd grpc-trt-fgpa

make

./server
