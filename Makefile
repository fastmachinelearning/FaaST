#note the compile for below needs to be pointed in the right direction
LDFLAGS = -Wl,-rpath,/usr/local/lib\
 -L/usr/local/lib \
 -lgrpc++_unsecure -lgrpc -lupb -lprotobuf -lpthread -lgrpc_unsecure -lgpr \
 -lcares -lgrpc_plugin_support -lssl -lcrypto  -lz -labsl_strings -labsl_base -labsl_throw_delegate \
 $(opencl_LDFLAGS)\
 -lnsl -laddress_sorting -ldl -lm

COMMON_REPO := ./hls4ml_c/

include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk
#include $(COMMON_REPO)/utility/rules.mk

HLS4ML_PROJ_TYPE := DENSE

CXX = g++
CPPFLAGS += `pkg-config --cflags protobuf grpc`
CPPFLAGS += -I$(CURDIR)/../../include -I$(COMMON_REPO)/src/ -I$(COMMON_REPO)/src/nnet_utils/ $(xcl2_CXXFLAGS) $(opencl_CXXFLAGS) -DIS_$(HLS4ML_PROJ_TYPE) -DWEIGHTS_DIR=$(COMMON_REPO)/src/weights/ -I$(XILINX_VIVADO)/include/ -I$(XILINX_SDACCEL)/include/ -Wno-unknown-pragmas
CXXFLAGS := -std=c++11


GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

all: server client

xcl2.o: $(xcl2_SRCS) $(xcl2_HDRS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

client: api.pb.o model_config.pb.o request_status.pb.o server_status.pb.o grpc_service.pb.o api.grpc.pb.o model_config.grpc.pb.o request_status.grpc.pb.o server_status.grpc.pb.o grpc_service.grpc.pb.o client.o
	$(CXX) $^ $(LDFLAGS) -o $@

server: api.pb.o model_config.pb.o request_status.pb.o server_status.pb.o grpc_service.pb.o api.grpc.pb.o model_config.grpc.pb.o request_status.grpc.pb.o server_status.grpc.pb.o grpc_service.grpc.pb.o server.o xcl2.o
	$(CXX) $^ $(LDFLAGS) -o $@

%.grpc.pb.cc: %.proto
	protoc --grpc_out=. --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) $<

%.pb.cc: %.proto
	protoc --cpp_out=. $<

clean:
	rm -f *.o *.pb.cc *.pb.h server
