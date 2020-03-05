#note the compile for below needs to be pointed in the right direction
LDFLAGS = -Wl,-rpath,/usr/local/lib\
 -L/usr/local/lib \
 -lgrpc++_unsecure -lgrpc -lupb -lprotobuf -lpthread -lgrpc_unsecure -lgpr \
 -lcares -lgrpc_plugin_support -lssl -lcrypto  -lz -labsl_strings -labsl_base -labsl_throw_delegate \
 -lnsl -laddress_sorting -ldl -lm

CXX = g++
CPPFLAGS += `pkg-config --cflags protobuf grpc`
CPPFLAGS += -I$(CURDIR)/../include
CXXFLAGS += -std=c++11

GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

all: server client

client: api.pb.o model_config.pb.o request_status.pb.o server_status.pb.o grpc_service.pb.o api.grpc.pb.o model_config.grpc.pb.o request_status.grpc.pb.o server_status.grpc.pb.o grpc_service.grpc.pb.o client.o
	$(CXX) $^ $(LDFLAGS) -o $@

server: api.pb.o model_config.pb.o request_status.pb.o server_status.pb.o grpc_service.pb.o api.grpc.pb.o model_config.grpc.pb.o request_status.grpc.pb.o server_status.grpc.pb.o grpc_service.grpc.pb.o server.o
	$(CXX) $^ $(LDFLAGS) -o $@

%.grpc.pb.cc: %.proto
	protoc --grpc_out=. --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) $<

%.pb.cc: %.proto
	protoc --cpp_out=. $<

clean:
	rm -f *.o *.pb.cc *.pb.h server
