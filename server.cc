#include <iostream>
#include <memory>
#include <string>
//#include <boost/compute.hpp>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <thread>
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::system_clock SClock;

#include "xcl2.hpp"
#include <vector>

#include <grpcpp/grpcpp.h>
#include <grpc/impl/codegen/status.h>
#include "grpc_service_v2.grpc.pb.h"
#include "model_config_v2.pb.h"

#include "kernel_params.h"

#define NUM_CU 4
#define NBUFFER 8

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;
using grpc::StatusCode;

using inference::GRPCInferenceService;
using inference::ModelInferRequest;
using inference::ModelInferResponse;
using inference::ModelConfigRequest;
using inference::ModelConfigResponse;
using inference::ModelMetadataRequest;
using inference::ModelMetadataResponse;
using inference::DataType;

class GRPCServiceImplementation final : public inference::GRPCInferenceService::Service {

 public:
  std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_in;
  std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_hw_results;
  cl::Program program;
  std::vector<cl::CommandQueue> q;
  std::vector<cl::Kernel> krnl_xil;
  int ikern;
  std::vector<bool> isFirstRun;
  std::vector<std::vector<cl::Event>>   writeList;
  std::vector<std::vector<cl::Event>>   kernList;
  std::vector<std::vector<cl::Event>>   readList;
  std::vector<cl::Buffer> buffer_in;
  std::vector<cl::Buffer> buffer_out;
  std::vector<cl::Event>   write_event;
  std::vector<cl::Event>   kern_event;
  std::mutex mtx;
  std::mutex mtxi_write[NUM_CU*NBUFFER];

  std::pair<int,bool> get_info_lock() {
    int i;
    bool first;
    mtx.lock();
    i = ikern++;
    first = isFirstRun[i];
    if (first) isFirstRun[i]=false;
    if (ikern==NUM_CU*NBUFFER) ikern=0;
    mtx.unlock();
    return std::make_pair(i,first);
  }
  void get_ilock_write(int ik) {
    mtxi_write[ik].lock();
  }
  void release_ilock_write(int ik) {
    mtxi_write[ik].unlock();
  }

  unsigned int verbosity = 0;

 private:

  cl_int err;

  void print_nanoseconds(std::string prefix, std::chrono::time_point<std::chrono::system_clock> now, int ik) {
    auto duration = now.time_since_epoch();
    
    typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<8>
    >::type> Days; /* UTC: +8:00 */
    
    Days days = std::chrono::duration_cast<Days>(duration);
        duration -= days;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= minutes;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        duration -= seconds;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        duration -= milliseconds;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        duration -= microseconds;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

    std::cout << "KERN" << ik << ", " << prefix << hours.count() << ":"
          << minutes.count() << ":"
          << seconds.count() << ":"
          << milliseconds.count() << ":"
          << microseconds.count() << ":"
          << nanoseconds.count() << std::endl;
  }

  grpc::Status ModelConfig(ServerContext* context,
                           const ModelConfigRequest* request,
                           ModelConfigResponse* reply) override {
    if (verbosity>1)
      std::cout<<"In Config"<<std::endl;
    if (request->name()=="facile_all_v2") {
      auto config = reply->mutable_config();
      config->set_name(request->name());
      config->set_max_batch_size(16384);
      return grpc::Status::OK;
    }
    else {
      return grpc::Status::CANCELLED;
    }
  }

  grpc::Status ModelMetadata(ServerContext* context,
                           const ModelMetadataRequest* request,
                           ModelMetadataResponse* reply) override {
    if (verbosity>1) 
      std::cout<<"In Metadata"<<std::endl;
    if (request->name()=="facile_all_v2") {
      std::string conv_prefix = "_DataConverter:";
      std::string converter = "FloatApFixed16F6Converter";
      reply->set_name(request->name());
      reply->add_versions(request->version());
      auto input = reply->add_inputs();
      input->set_name("input");
      input->set_name(input->name()+conv_prefix+converter); //designed to allow passing of desired data converter
      input->set_datatype("UINT16");
      input->add_shape(-1);
      input->add_shape(32);
      auto output = reply->add_outputs();
      output->set_name("output");
      output->set_name(output->name()+conv_prefix+converter); //designed to allow passing of desired data converter
      output->set_datatype("UINT16");
      output->add_shape(-1);
      output->add_shape(1);
      return grpc::Status::OK;
    }
    else {
      return grpc::Status::CANCELLED;
    }
  }

  grpc::Status ModelInfer(
		     ServerContext* context, 
		     const ModelInferRequest* request, 
		     ModelInferResponse* reply
		     ) override {
    if (verbosity>1) 
      std::cout<<"In Infer"<<std::endl;
    auto t0 = Clock::now();
    auto ikf = get_info_lock();
    int ikb = ikf.first;
    int ik = ikb%NUM_CU;
    bool firstRun = ikf.second;
    auto ts1_ = SClock::now();
    if (verbosity>2) {
      std::cout<<"Running kernel "<<ik<<"... first run? ("<<firstRun<<")"<<std::endl;
      print_nanoseconds("   in Infer  ",ts1_, ikb);
    }
    const std::string& raw = request->raw_input_contents(0);
    const void* lVals = raw.c_str();
    data_t* lFVals = (data_t*) lVals;
    unsigned batch_size = raw.size()/32/sizeof(data_t);

    reply->set_id(request->id());
    reply->set_model_version("");

    //setup output (this is critical)
    auto output1 = reply->add_outputs();
    output1->set_name("output");


    auto ts1p = SClock::now();
    if (verbosity>2)
      print_nanoseconds("   pre-lock  ",ts1p, ikb);
    get_ilock_write(ikb);
    auto ts1 = SClock::now();
    if (verbosity>2)
      print_nanoseconds("       start:   ",ts1, ik);
    memcpy(source_in.data()+ikb*STREAMSIZE, &lFVals[0], batch_size*sizeof(bigdata_t));

    auto t1 = Clock::now();
    auto ts1a = SClock::now();
    if (verbosity>2)
      print_nanoseconds("   memcpy  ",ts1a, ikb);
    if (!firstRun) {
        OCL_CHECK(err, err = kern_event[ikb].wait());
    }
    auto ts1b = SClock::now();
    if (verbosity>2)
      print_nanoseconds("   kernwait  ",ts1b, ikb);
    OCL_CHECK(err,
              err =
                  q[ik].enqueueMigrateMemObjects({buffer_in[ikb]},
                                             0,
                                             NULL,
                                             &(write_event[ikb])));
    auto ts1c = SClock::now();
    if (verbosity>2)
      print_nanoseconds("       write:   ",ts1c, ik);
    
    writeList[ikb].clear();
    writeList[ikb].push_back(write_event[ikb]);
    //Launch the kernel
    OCL_CHECK(err,
              err = q[ik].enqueueNDRangeKernel(
                  krnl_xil[ikb], 0, 1, 1, &(writeList[ikb]), &(kern_event[ikb])));
    auto ts1d = SClock::now();
    if (verbosity>2)
      print_nanoseconds("      kernel:   ",ts1d, ik);
    kernList[ikb].clear();
    kernList[ikb].push_back(kern_event[ikb]);
    cl::Event read_event;
    OCL_CHECK(err,
              err = q[ik].enqueueMigrateMemObjects({buffer_out[ikb]},
                                               CL_MIGRATE_MEM_OBJECT_HOST,
                                               &(kernList[ikb]),
                                               &(read_event)));
    auto ts1e = SClock::now();
    if (verbosity>2)
      print_nanoseconds("        read:   ",ts1e, ik);

    release_ilock_write(ikb);
    OCL_CHECK(err, err = kern_event[ikb].wait());
    OCL_CHECK(err, err = read_event.wait());
    auto ts1f = SClock::now();
    if (verbosity>2)
      print_nanoseconds("   readwait  ",ts1f, ikb);
    auto t2 = Clock::now();
    auto ts2 = SClock::now();
    if (verbosity>2) {
      print_nanoseconds("       finish:  ",ts2, ik);
      std::cout << " FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" << std::endl;
    }

    //Finally deal with the ouputs
    std::string *outputs1 = reply->add_raw_output_contents();
    char* lTVals = new char[batch_size*sizeof(data_t)];
    memcpy(&lTVals[0], source_hw_results.data()+(ikb*COMPSTREAMSIZE), (batch_size)*sizeof(data_t));
    outputs1->append(lTVals,(batch_size)*sizeof(data_t));

    delete[] lTVals;

    auto ts1g = SClock::now();
    if (verbosity>2)
      print_nanoseconds("   finish  ",ts1g, ikb);

    auto t3 = Clock::now();
    if (verbosity>2) {
      std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count() << " ns" << std::endl;
      std::cout << "   T1 time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << " ns" << std::endl;
      std::cout << "   T2 time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() << " ns" << std::endl;
      std::cout << " FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" << std::endl;
    }

    return grpc::Status::OK;
  } 
};

void runWait(std::unique_ptr<Server> srv) {
    srv->Wait();
}

void Run(std::string xclbinFilename, int port, unsigned int num_servers, unsigned int verbosity) {
  GRPCServiceImplementation service;
  service.verbosity = verbosity;

  ServerBuilder builders[num_servers];
  for (unsigned int is = 0; is < num_servers; is++) {
      std::string srv_address = "0.0.0.0:"+std::to_string(port+is);
      builders[is].AddListeningPort(srv_address, grpc::InsecureServerCredentials());
      builders[is].SetMaxMessageSize(10000000);
      builders[is].RegisterService(&service);
  }

  size_t vector_size_in_bytes  = sizeof(bigdata_t) * STREAMSIZE;
  size_t vector_size_out_bytes = sizeof(bigdata_t) * COMPSTREAMSIZE;
  // Allocate Memory in Host Memory
  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
  // is used if it is properly aligned. when not aligned, runtime had no choice but to create
  // its own host side buffer. So it is recommended to use this allocator if user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
  // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 

  //initialize
  service.source_in.reserve(STREAMSIZE*NUM_CU*NBUFFER);
  service.source_hw_results.reserve(COMPSTREAMSIZE*NUM_CU*NBUFFER);

  std::vector<cl::Device> devices = xcl::get_xil_devices();
  cl::Device device = devices[0];
  devices.resize(1);

  cl::Context context(device);
  for (int ik = 0; ik < NUM_CU; ik++) {
    cl::CommandQueue q_tmp(context, device,  CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    service.q.push_back(q_tmp);
  }
  std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 
  if (verbosity>0)
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

  cl::Program::Binaries bins;
  // Load xclbin
  if (xclbinFilename != "") {
    if (verbosity>0)
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
  for (int ib = 0; ib < NBUFFER; ib++) {
    for (int ik = 0; ik < NUM_CU; ik++) {
      std::string kernel_name = "alveo_hls4ml:{alveo_hls4ml_" + std::to_string(ik) + "}";
      cl::Kernel krnl_alveo_hls4ml(service.program,kernel_name.c_str());
      service.krnl_xil.push_back(krnl_alveo_hls4ml);
    }
  }

  service.writeList.reserve(NUM_CU*NBUFFER);
  service.kernList.reserve(NUM_CU*NBUFFER);
  service.readList.reserve(NUM_CU*NBUFFER);
  for (int ib = 0; ib < NBUFFER; ib++) {
    for (int ik = 0; ik < NUM_CU; ik++) {
      // Allocate Buffer in Global Memory
      // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
      // Device-to-host communication
      cl::Buffer buffer_in_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_bytes, service.source_in.data()+((ib*NUM_CU+ik) * STREAMSIZE));
      cl::Buffer buffer_out_tmp(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_size_out_bytes, service.source_hw_results.data()+((ib*NUM_CU+ik) * COMPSTREAMSIZE));
      service.buffer_in.push_back(buffer_in_tmp);
      service.buffer_out.push_back(buffer_out_tmp);
  
      cl::Event tmp_write = cl::Event();
      cl::Event tmp_kern = cl::Event();
      cl::Event tmp_read = cl::Event();
      service.write_event.push_back(tmp_write);
      service.kern_event.push_back(tmp_kern);
  
      int narg = 0;
      service.krnl_xil[ib*NUM_CU+ik].setArg(narg++, service.buffer_in[ib*NUM_CU+ik]);
      service.krnl_xil[ib*NUM_CU+ik].setArg(narg++, service.buffer_out[ib*NUM_CU+ik]);
      service.isFirstRun.push_back(true);
      std::vector<cl::Event> tmp_write_vec(1);
      std::vector<cl::Event> tmp_kern_vec(1);
      std::vector<cl::Event> tmp_read_vec(1);
      service.writeList.push_back(tmp_write_vec);
      service.kernList.push_back(tmp_kern_vec);
      service.readList.push_back(tmp_read_vec);
    }
  }

  service.ikern = 0;
  std::vector<std::thread> th_vec;

  for (unsigned int it = 0; it < num_servers; it++) {
      std::unique_ptr<Server> server(builders[it].BuildAndStart());
      std::thread th(runWait, std::move(server));
      th_vec.push_back(std::move(th));
      if (verbosity>0) 
        std::cout << "Server listening on port: " << port+it << std::endl;
  }
  for (std::thread & th : th_vec)
  {
      if (th.joinable())
      th.join();
  }
}

int main(int argc, char** argv) {


  std::string xclbinFilename = "";
  int port = 5001;
  unsigned int num_servers = 1;
  unsigned int verbosity = 0;
  if (argc>1) xclbinFilename = argv[1];
  if (argc>2) port = std::atoi(argv[2]);
  if (argc>3) num_servers = std::atoi(argv[3]);
  if (argc>4) verbosity = std::atoi(argv[4]);
  Run(xclbinFilename,port,num_servers,verbosity);

  return 0;
}
