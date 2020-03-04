#note the compile for below needs to be pointed in the right direction
LDFLAGS = -Wl,-rpath,/usr/local/lib\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/libgrpc++_unsecure.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/libgrpc.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/libupb.a\
 /usr/local/lib/libprotobuf.a\
 -lpthread /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/libgrpc_unsecure.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/libgpr.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/cares/cares/lib/libcares.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/libgrpc_plugin_support.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/boringssl-with-bazel/libssl.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/boringssl-with-bazel/libcrypto.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/zlib/libz.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/strings/libabsl_strings.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/base/libabsl_base.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/base/libabsl_throw_delegate.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/base/libabsl_log_severity.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/base/libabsl_raw_logging_internal.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/hash/libabsl_hash.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/time/libabsl_time.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/time/libabsl_civil_time.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/time/libabsl_time_zone.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/strings/libabsl_str_format_internal.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/strings/libabsl_strings_internal.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/types/libabsl_bad_optional_access.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/types/libabsl_bad_any_cast_impl.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/types/libabsl_bad_variant_access.a\
 /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/third_party/abseil-cpp/absl/numeric/libabsl_int128.a\
 -lnsl /storage/local/data1/home/drankin/HCAL_GRPC/grpc/cmake/build/libaddress_sorting.a\
 -ldl -lrt -lm 
 #/storage/local/data1/home/drankin/HCAL_GRPC/local/protobuf-trt/lib64/libprotobuf-trt.a\
 #/usr/local/lib/libcares.so.2.3.0\


CXX = g++
CPPFLAGS += `pkg-config --cflags protobuf grpc`
CPPFLAGS += -I/storage/local/data1/home/drankin/HCAL_GRPC/grpc/include/
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
