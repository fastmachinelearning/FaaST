#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "grpc_service.grpc.pb.h"
#include "request_status.grpc.pb.h"
#include "model_config.pb.h"

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
    
    //Fnally deal with the ouputs
    float* lTVals = new float[1];
    for(int i0 = 0; i0 < 1; i0++) lTVals[i0] = 6.;
    char* tmp = (char*) lTVals;
    std::string *outputs1 = reply->add_raw_output();
    for(unsigned i0 = 0; i0 < batch_size; i0++) { 
      outputs1->append(tmp,sizeof(tmp));
    }
    return grpc::Status::OK;
  } 
};

void Run() {
  std::string address("0.0.0.0:8080");
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
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on port: " << address << std::endl;
  int server_id=1;
  server->Wait();
}

int main(int argc, char** argv) {
  Run();

  return 0;
}
