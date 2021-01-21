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
#include "grpc_service_v2.grpc.pb.h"

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

using inference::GRPCInferenceService;
using inference::ModelInferRequest;
using inference::ModelInferResponse;

class GRPCInferenceServiceClient {
public:
  GRPCInferenceServiceClient(std::shared_ptr<Channel> channel) : stub_(GRPCInferenceService::NewStub(channel)) {}

  int ntest;

  data_t* lTVals;

  std::string sendRequest(int a, int b) {

    char *tmp = (char*) lTVals;
    std::string tmp2 = "Success";

    ModelInferRequest request;
    request.set_model_name("facile_v2_all");
    request.set_model_version("");

    std::string* new_input = request.add_raw_input_contents();
    new_input->append(tmp, sizeof(data_t)*32*STREAMSIZE);

    for (int it = 0; it < ntest; it++) {
      grpc::ClientContext context;
      ModelInferResponse response;
      grpc::Status status = stub_->ModelInfer(&context,request,&response);
      bool ok = status.ok();
      if(ok){
      } else {
        std::cout<<"Failed"<<std::endl;
        return "Fail";
      }
    }
    return tmp2;
  }

private:
  std::unique_ptr<GRPCInferenceService::Stub> stub_;
};

void submit(GRPCInferenceServiceClient& client) {
    int a = 5; int b = 10;
    std::string response;
    client.sendRequest(a,b);
    std::string &raw = response;
    const void* lVals = raw.c_str();
    bigdata_t* lFVals = (bigdata_t*) lVals;
}

void Run(int ntest, int nthreads, int port) {
  std::string address_base("localhost:");
  std::string address = address_base + std::to_string(port);
  std::vector<GRPCInferenceServiceClient> clients;
  for (int it = 0; it < nthreads; it++) {
      ChannelArguments args;
      args.SetMaxSendMessageSize(100000000+it);
      GRPCInferenceServiceClient client(grpc::CreateCustomChannel(
          address, grpc::InsecureChannelCredentials(), args));

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

