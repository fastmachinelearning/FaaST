#note the compile for below needs to be pointed in the right direction
LDFLAGS = -Wl,-rpath,/usr/local/lib /usr/local/lib/libgrpc++_unsecure.a /usr/local/lib64/libprotobuf.a -lpthread /usr/local/lib/libgrpc_unsecure.a /usr/local/lib/libgpr.a /usr/local/lib/libz.so /usr/local/lib/libcares.so.2.3.0 -lnsl /usr/local/lib/libaddress_sorting.a -ldl -lrt -lm 

CXX = g++
CPPFLAGS += `pkg-config --cflags protobuf grpc`
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
