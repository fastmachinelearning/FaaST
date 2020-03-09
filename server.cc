#include <iostream>
#include <memory>
#include <string>

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#include "xcl2.hpp"
#include <vector>

#include <grpcpp/grpcpp.h>
#include "grpc_service.grpc.pb.h"
#include "request_status.grpc.pb.h"
#include "model_config.pb.h"

#include "kernel_params.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using nvidia::inferenceserver::GRPCService;
using nvidia::inferenceserver::InferRequest;
using nvidia::inferenceserver::InferResponse;
using nvidia::inferenceserver::StatusRequest;
using nvidia::inferenceserver::StatusResponse;
using nvidia::inferenceserver::RequestStatusCode;
using nvidia::inferenceserver::DataType;

class GRPCServiceImplementation final : public nvidia::inferenceserver::GRPCService::Service {

 public:
  std::vector<data_t,aligned_allocator<data_t>> source_in;
  std::vector<data_t,aligned_allocator<data_t>> source_hw_results;
//(DATA_SIZE_IN*STREAMSIZE);
//(DATA_SIZE_OUT*STREAMSIZE);
  std::vector<cl::Memory> inBufVec, outBufVec;
  cl::Kernel krnl_xil;
  cl::Program program;
  cl::CommandQueue q;

 private:
  grpc::Status Status(
		     ServerContext* context, 
		     const StatusRequest* request, 
		     StatusResponse* reply
		     ) override {

    //std::cout << "In Status" << std::endl;
    auto server_status = reply->mutable_server_status();
    server_status->set_id("inference:0");
    auto& model_status  = *server_status->mutable_model_status();
    auto config = model_status["facile"].mutable_config();
    config->set_max_batch_size(160000);
    //Hcal config
    config->set_name("facile");
    auto input = config->add_input();
    input->set_name("input");
    input->set_data_type(DataType::TYPE_FP32);
    input->add_dims(15);
    auto output = config->add_output();
    output->set_name("output/BiasAdd");
    output->set_data_type(DataType::TYPE_FP32);
    output->add_dims(1);
    reply->mutable_request_status()->set_code(RequestStatusCode::SUCCESS);
    nvidia::inferenceserver::RequestStatus request_status = reply->request_status();
    nvidia::inferenceserver::ServerStatus  check_server_status  = reply->server_status();
    return grpc::Status::OK;
  }

  grpc::Status Infer(
		     ServerContext* context, 
		     const InferRequest* request, 
		     InferResponse* reply
		     ) override {
    
    //std::cout << "In Infer" << std::endl;
    const std::string& raw = request->raw_input(0);
    const void* lVals = raw.c_str();
    float* lFVals = (float*) lVals;
    //output array that is equal to ninputs(15)*batch flot is 4 bits
    unsigned batch_size = raw.size()/15/4;
    reply->mutable_request_status()->set_code(RequestStatusCode::SUCCESS);
    reply->mutable_request_status()->set_server_id("inference:0");
    reply->mutable_meta_data()->set_id(request->meta_data().id());
    reply->mutable_meta_data()->set_model_version(-1);
    reply->mutable_meta_data()->set_batch_size(batch_size);

    //setup output (this is critical)
    auto output1 = reply->mutable_meta_data()->add_output();
    output1->set_name("output/BiasAdd");
    output1->mutable_raw()->mutable_dims()->Add(1);
    output1->mutable_raw()->set_batch_byte_size(8*batch_size);

    auto t1 = Clock::now();
    auto t2 = Clock::now();

    std::vector<float> pr;
    for (int istream = 0; istream < (batch_size < STREAMSIZE ? batch_size : STREAMSIZE); istream++) {
        // Create the test data if no data files found or if end of files has been reached
        for(int j = 0 ; j < DATA_SIZE_IN; j++){
            source_in[istream*DATA_SIZE_IN+j] = (data_t)(lFVals[istream*DATA_SIZE_IN+j]);
        }
        for(int j = 0 ; j < DATA_SIZE_OUT*STREAMSIZE ; j++){
            source_hw_results[j] = 0;
        }
    }

    t1 = Clock::now();
    // Copy input data to device global memory
    q.enqueueMigrateMemObjects(inBufVec,0/* 0 means from host*/);
    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is recommended
    // to always use enqueueTask() for invoking HLS kernel
    q.enqueueTask(krnl_xil);
    // Copy Result from Device Global Memory to Host Local Memory
    q.enqueueMigrateMemObjects(outBufVec,CL_MIGRATE_MEM_OBJECT_HOST);
    // Check for any errors from the command queue
    q.finish();
    t2 = Clock::now();
    std::cout << "FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" << std::endl;

    //std::cout<<"Quantized predictions: \n";
    //for (int j = 0; j < (batch_size < STREAMSIZE ? batch_size : STREAMSIZE); j++) {
    //    for (int k = 0 ; k < DATA_SIZE_OUT ; k++){
    //        std::cout << source_hw_results[j*DATA_SIZE_OUT + k] << " \t";
    //    }
    //}
    //std::cout << source_hw_results[1];
    //std::cout << std::endl;
    
    //Finally deal with the ouputs
    std::string *outputs1 = reply->add_raw_output();
    for(unsigned i0 = 0; i0 < batch_size; i0++) { 
      float* lTVals = new float[1];
      lTVals[0] = source_hw_results[i0];
      char* tmp = (char*) lTVals;
      outputs1->append(tmp,sizeof(tmp));
    }
    return grpc::Status::OK;
  } 
};

void Run(std::string xclbinFilename) {
  std::string address("0.0.0.0:8081");
  GRPCServiceImplementation service;

  ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.SetMaxMessageSize(640000);
  builder.RegisterService(&service);
  //All the crap that trt inference server runs
  //std::unique_ptr<grpc::ServerCompletionQueue> health_cq = builder.AddCompletionQueue();
  //std::unique_ptr<grpc::ServerCompletionQueue> status_cq = builder.AddCompletionQueue();
  //std::unique_ptr<grpc::ServerCompletionQueue> repository_cq   = builder.AddCompletionQueue();
  //std::unique_ptr<grpc::ServerCompletionQueue> infer_cq        = builder.AddCompletionQueue();
  //std::unique_ptr<grpc::ServerCompletionQueue> stream_infer_cq = builder.AddCompletionQueue();
  //std::unique_ptr<grpc::ServerCompletionQueue> modelcontrol_cq = builder.AddCompletionQueue();
  //std::unique_ptr<grpc::ServerCompletionQueue> shmcontrol_cq   = builder.AddCompletionQueue();

  size_t vector_size_in_bytes = sizeof(data_t) * DATA_SIZE_IN * STREAMSIZE;
  size_t vector_size_out_bytes = sizeof(data_t) * DATA_SIZE_OUT * STREAMSIZE;
  // Allocate Memory in Host Memory
  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
  // is used if it is properly aligned. when not aligned, runtime had no choice but to create
  // its own host side buffer. So it is recommended to use this allocator if user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
  // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 

  //initialize
  for(int j = 0 ; j < DATA_SIZE_IN*STREAMSIZE ; j++){
      service.source_in.push_back((unsigned int)0);
  }
  for(int j = 0 ; j < DATA_SIZE_OUT*STREAMSIZE ; j++){
      service.source_hw_results.push_back((unsigned int)0);
  }

  std::vector<cl::Device> devices = xcl::get_xil_devices();
  cl::Device device = devices[0];
  devices.resize(1);

  cl::Context context(device);
  cl::CommandQueue q_tmp(context, device, CL_QUEUE_PROFILING_ENABLE);
  service.q = q_tmp;
  std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 
  std::cout << "Found Device=" << device_name.c_str() << std::endl;

  cl::Program::Binaries bins;
  // Load xclbin
  if (xclbinFilename != "") {
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    
    // Creating Program from Binary File
    bins.push_back({buf,nb});
  } else {
    // find_binary_file() is a utility API which will search the xclbin file for
    // targeted mode (sw_emu/hw_emu/hw) and for targeted platforms.
    std::string binaryFile = xcl::find_binary_file(device_name,"alveo_hls4ml");

    // import_binary_file() ia a utility API which will load the binaryFile
    // and will return Binaries.
    bins = xcl::import_binary_file(binaryFile);
  }
  cl::Program tmp_program(context, devices, bins);
  service.program = tmp_program;

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
  // Device-to-host communication
  cl::Buffer buffer_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
          vector_size_in_bytes, service.source_in.data());
  cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
          vector_size_out_bytes, service.source_hw_results.data());

  service.inBufVec.push_back(buffer_in);
  service.outBufVec.push_back(buffer_output);

  cl::Kernel krnl_alveo_hls4ml(service.program,"alveo_hls4ml");

  int narg = 0;
  krnl_alveo_hls4ml.setArg(narg++, buffer_in);
  krnl_alveo_hls4ml.setArg(narg++, buffer_output);
  service.krnl_xil = krnl_alveo_hls4ml;

  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on port: " << address << std::endl;
  int server_id=1;
  server->Wait();
}

int main(int argc, char** argv) {
  std::string xclbinFilename = "";
  if (argc>1) xclbinFilename = argv[1];
  Run(xclbinFilename);

  return 0;
}
