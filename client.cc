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
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::system_clock SClock;

#include <grpcpp/grpcpp.h>
#include "grpc_service.grpc.pb.h"

#include "xcl2.hpp"
#include <vector>
#include "kernel_params.h"

#include <thread>

#ifdef STREAMSIZE
#undef STREAMSIZE
#endif
#define STREAMSIZE 16384

using grpc::Channel;
using grpc::ChannelArguments;
using grpc::ClientContext;
using grpc::Status;

using nvidia::inferenceserver::GRPCService;
using nvidia::inferenceserver::InferRequest;
using nvidia::inferenceserver::InferResponse;

class GRPCServiceClient {
public:
  GRPCServiceClient(std::shared_ptr<Channel> channel) : stub_(GRPCService::NewStub(channel)) {}

  int ntest;

  data_t* lTVals;

  std::string sendRequest(int a, int b) {

    char *tmp = (char*) lTVals;
    std::string tmp2 = "Success";

    nvidia::inferenceserver::InferRequest request;
    nvidia::inferenceserver::InferRequestHeader infer_request;
    infer_request.mutable_input()->Clear();
    infer_request.set_id(0);
    auto rinput = infer_request.add_input();
    rinput->set_name("facile");
    rinput->add_dims(32);
    rinput->set_batch_byte_size(sizeof(data_t)*32*STREAMSIZE);
    request.Clear();
    request.set_model_name("facile");
    request.set_model_version(-1);

    request.mutable_meta_data()->MergeFrom(infer_request);
    std::string* new_input = request.add_raw_input();
    new_input->append(tmp, sizeof(data_t)*32*STREAMSIZE);

    for (int it = 0; it < ntest; it++) {
      grpc::ClientContext context;
      InferResponse response;
      //context.set_compression_algorithm(GRPC_COMPRESS_DEFLATE);
      grpc::Status status = stub_->Infer(&context,request,&response);
      bool ok = status.ok();
    //std::cout<<status.error_message()<<std::endl;
      if(ok){
        //uintptr_t run_index = response.meta_data().id();
        //tmp2 = response.raw_output(0);
      } else {
        std::cout<<"Failed"<<std::endl;
        return "Fail";
      }
    }
    return tmp2;
  }

private:
  std::unique_ptr<GRPCService::Stub> stub_;
};

void submit(GRPCServiceClient& client) {
    int a = 5; int b = 10;
    std::string response;
    client.sendRequest(a,b);
    std::string &raw = response;
    const void* lVals = raw.c_str();
    bigdata_t* lFVals = (bigdata_t*) lVals;
    //std::cout << "Answer received: " << a << " * " << b << " = " << lFVals[0] << std::endl;
}

void Run(int ntest, int nthreads, int port) {
  //std::string address_base("localhost:");
  std::string address_base("34.220.173.87:");
  //std::string address_base("54.203.154.29:");
  std::string address = address_base + std::to_string(port);
  //std::string address("ailab01.fnal.gov:8001");
  // Set the default compression algorithm for the channel.
  //args.SetCompressionAlgorithm(GRPC_COMPRESS_DEFLATE);
  std::vector<GRPCServiceClient> clients;
  for (int it = 0; it < nthreads; it++) {
      ChannelArguments args;
      args.SetMaxSendMessageSize(100000000+it);
      GRPCServiceClient client(grpc::CreateCustomChannel(
          address, grpc::InsecureChannelCredentials(), args));
      /*GRPCServiceClient client(
                        grpc::CreateChannel(
                                            address, 
                                            grpc::InsecureChannelCredentials()
                                            )
                        );*/

      client.ntest = ntest;
      client.lTVals = new data_t[32*STREAMSIZE];
      for(int i0 = 0; i0 < 32*STREAMSIZE; i0++) client.lTVals[i0] = rand();
      clients.push_back(std::move(client));
  }

  std::cout<<"Running "<<nthreads<<" threads "<<ntest<<" times each..."<<std::endl;
  std::vector<std::thread> th_vec;
  auto t0 = Clock::now();
  for (int it = 0; it < nthreads; it++) {
      std::thread th(submit, std::ref(clients[it]));
      th_vec.push_back(std::move(th));
  }
  for (std::thread & th : th_vec)
  {
      if (th.joinable())
      th.join();
  }
  auto t3 = Clock::now();
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count() << " ns" << std::endl;
  std::cout << "Time per inference: " << float(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count()/1000000.) / float(ntest*nthreads) << " ms" << std::endl;
  std::cout << "Throughput: " << float(ntest*nthreads) / float(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count()/1000000000.) << " inf/s" << std::endl;
}

int main(int argc, char** argv){

  int ntest = 1;
  int nthreads = 1;
  int port = 5001;
  if (argc>1) ntest = std::atoi(argv[1]);
  if (argc>2) nthreads = std::atoi(argv[2]);
  if (argc>3) port = std::atoi(argv[3]);
  Run(ntest,nthreads,port);
  return 0;
}

