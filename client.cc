#include <string>

#include <grpcpp/grpcpp.h>
#include "grpc_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using nvidia::inferenceserver::GRPCService;
using nvidia::inferenceserver::InferRequest;
using nvidia::inferenceserver::InferResponse;

class GRPCServiceClient {
public:
  GRPCServiceClient(std::shared_ptr<Channel> channel) : stub_(GRPCService::NewStub(channel)) {}

  const std::string sendRequest(int a, int b) {
    nvidia::inferenceserver::InferRequest request;
    nvidia::inferenceserver::InferRequestHeader infer_request;
    infer_request.mutable_input()->Clear();
    infer_request.set_id(0);
    auto rinput = infer_request.add_input();
    rinput->set_name("facile");
    rinput->add_dims(15);
    rinput->set_batch_byte_size(1024);
    request.Clear();
    request.set_model_name("facile");
    request.set_model_version(0);
    request.mutable_meta_data()->MergeFrom(infer_request); 

    float* lTVals = new float[15];
    for(int i0 = 0; i0 < 15; i0++) lTVals[i0] = 6.;
    char* tmp = (char*) lTVals;
    std::string* new_input = request.add_raw_input();
    new_input->append(tmp, sizeof(tmp));
    grpc::ClientContext context;
    InferResponse response;   
    grpc::Status status = stub_->Infer(&context,request,&response);
    bool ok = status.ok();
    if(ok){
      uintptr_t run_index = response.meta_data().id();
      std::string tmp2 = response.raw_output(0);
      return tmp2;
    } else {
      return "Fail";
    }
  }

private:
  std::unique_ptr<GRPCService::Stub> stub_;
};

void Run() {
  std::string address("localhost:8001");
  //std::string address("ailab01.fnal.gov:8001");
  GRPCServiceClient client(
			grpc::CreateChannel(
					    address, 
					    grpc::InsecureChannelCredentials()
					    )
			);

  std::string response;
  int a = 5;
  int b = 10;
  response = client.sendRequest(a, b);
  std::cout << "Answer received: " << a << " * " << b << " = " << response << std::endl;
}

int main(int argc, char* argv[]){
  Run();
  return 0;
}
