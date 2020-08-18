/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <memory>
#include <string>
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::system_clock SClock;

#include "xcl2.hpp"
#include <vector>
#include "kernel_params.h"

#include <thread>
#include <sstream>

#define NUM_CU 4
#define NBUFFER 8

#define STRINGIFY2(var) #var
#define STRINGIFY(var) STRINGIFY2(var)

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

void print_nanoseconds(std::string prefix, std::chrono::time_point<std::chrono::system_clock> now, int ik, std::stringstream &ss) {
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

    ss << "KERN" << ik << ", " << prefix << hours.count() << ":"
          << minutes.count() << ":"
          << seconds.count() << ":"
          << milliseconds.count() << ":"
          << microseconds.count() << ":"
          << nanoseconds.count() << "\n";
}

// An event callback function that prints the operations performed by the OpenCL
// runtime.
void event_cb(cl_event event1, cl_int cmd_status, void *data) {
    cl_int err;
    cl_command_type command;
    cl::Event event(event1, true);
    OCL_CHECK(err, err = event.getInfo(CL_EVENT_COMMAND_TYPE, &command));
    cl_int status;
    OCL_CHECK(err,
              err = event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status));
    const char *command_str;
    const char *status_str;
    switch (command) {
    case CL_COMMAND_READ_BUFFER:
        command_str = "buffer read";
        break;
    case CL_COMMAND_WRITE_BUFFER:
        command_str = "buffer write";
        break;
    case CL_COMMAND_NDRANGE_KERNEL:
        command_str = "kernel";
        break;
    case CL_COMMAND_MAP_BUFFER:
        command_str = "kernel";
        break;
    case CL_COMMAND_COPY_BUFFER:
        command_str = "kernel";
        break;
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
        command_str = "buffer migrate";
        break;
    default:
        command_str = "unknown";
    }
    switch (status) {
    case CL_QUEUED:
        status_str = "Queued";
        break;
    case CL_SUBMITTED:
        status_str = "Submitted";
        break;
    case CL_RUNNING:
        status_str = "Executing";
        break;
    case CL_COMPLETE:
        status_str = "Completed";
        break;
    }
    printf("[%s]: %s %s\n",
           reinterpret_cast<char *>(data),
           status_str,
           command_str);
    fflush(stdout);
}

// Sets the callback for a particular event
void set_callback(cl::Event event, const char *queue_name) {
    cl_int err;
    OCL_CHECK(err,
              err =
                  event.setCallback(CL_COMPLETE, event_cb, (void *)queue_name));
}

class fpgaObj {
  public:
    std::stringstream ss;
    int ithr;
    int nevents;
    int ikern;
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_in;
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_hw_results;
    cl::Program program;
    std::vector<cl::CommandQueue> q;
    std::vector<cl::Kernel> krnl_xil;
    std::vector<std::vector<cl::Event>>   writeList;
    std::vector<std::vector<cl::Event>>   kernList;
    std::vector<std::vector<cl::Event>>   readList;
    std::vector<cl::Buffer> buffer_in;
    std::vector<cl::Buffer> buffer_out;
    std::vector<cl::Event>   write_event;
    std::vector<cl::Event>   kern_event;
    std::vector<bool> isFirstRun;
    cl_int err;

    std::pair<int,bool> get_info_lock() {
      int i;
      bool first;
      mtx.lock();
      i = ikern++;
      if (ikern==NUM_CU*NBUFFER) ikern = 0;
      first = isFirstRun[i];
      if (first) isFirstRun[i]=false;
      mtx.unlock();
      return std::make_pair(i,first);
    }
    void get_ilock(int ik) {
      mtxi[ik].lock();
    }
    void release_ilock(int ik) {
      mtxi[ik].unlock();
    }
    void write_ss_safe(std::string newss) {
      smtx.lock();
      ss << "Thread " << ithr << "\n" << newss << "\n";
      ithr++;
      smtx.unlock();
    }

    std::stringstream runFPGA() {
        auto t0 = Clock::now();
        auto t1 = Clock::now();
        auto t1a = Clock::now();
        auto t1b = Clock::now();
        auto t2 = Clock::now();
        auto t3 = Clock::now();
        std::stringstream ss;
        unsigned int lFVals[16*STREAMSIZE];
        for (unsigned int j = 0; j < 16*STREAMSIZE; j++) {
            lFVals[j] = rand();
        }
    
        for (int i = 0 ; i < nevents ; i++){
            t0 = Clock::now();
            auto ikf = get_info_lock();
            int ikb = ikf.first;
            int ik = ikb%NUM_CU;
            bool firstRun = ikf.second;

            memcpy(source_in.data()+ikb*STREAMSIZE, &lFVals[0], STREAMSIZE*sizeof(bigdata_t));
    
            t1 = Clock::now();
            auto ts1 = SClock::now();
            print_nanoseconds("        start:  ",ts1, ik, ss);
            std::string queuename = "ooo_queue "+std::to_string(ikb);
        
            get_ilock(ikb);
            //Copy input data to device global memory
            if (!firstRun) {
                OCL_CHECK(err, err = kern_event[ikb].wait());
            }
            OCL_CHECK(err,
                      err =
                          q[ik].enqueueMigrateMemObjects({buffer_in[ikb]},
                                                     0 /* 0 means from host*/,
                                                     NULL,
                                                     &(write_event[ikb])));
            //set_callback(write_event[ikb], queuename.c_str());
    
            t1a = Clock::now();
            writeList[ikb].clear();
            writeList[ikb].push_back(write_event[ikb]);
            //Launch the kernel
            OCL_CHECK(err,
                      err = q[ik].enqueueNDRangeKernel(
                          krnl_xil[ikb], 0, 1, 1, &(writeList[ikb]), &(kern_event[ikb])));
            //set_callback(kern_event[ikb], queuename.c_str());
            t1b = Clock::now();
            kernList[ikb].clear();
            kernList[ikb].push_back(kern_event[ikb]);
            cl::Event read_event;
            OCL_CHECK(err,
                      err = q[ik].enqueueMigrateMemObjects({buffer_out[ikb]},
                                                       CL_MIGRATE_MEM_OBJECT_HOST,
                                                       &(kernList[ikb]),
                                                       &(read_event)));

            //set_callback(read_event, queuename.c_str());
            release_ilock(ikb);
        
            OCL_CHECK(err, err = kern_event[ikb].wait());
            OCL_CHECK(err, err = read_event.wait());
            auto ts2 = SClock::now();
            print_nanoseconds("       finish:  ",ts2, ik, ss);
            t2 = Clock::now();
    
            t3 = Clock::now();
            //std::cout << " Prep time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << " ns" << std::endl;
            //std::cout << " FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" << std::endl;
            //std::cout << "    inputs: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1a - t1).count() << " ns" << std::endl;
            //std::cout << "    kernel: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1b - t1a).count() << " ns" << std::endl;
            //std::cout << "   outputs: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1b).count() << " ns" << std::endl;
            ss << "KERN"<<ik<<"   Total time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count() << " ns\n";
            //std::cout<<"---- END EVENT "<<i+1<<" ----"<<std::endl;
        }
        return ss;
    }

  private:
    mutable std::mutex mtx;
    mutable std::mutex mtxi[NUM_CU*NBUFFER];
    mutable std::mutex smtx;
};

void FPGA(fpgaObj& theFPGA) {
    std::stringstream ss;
    ss << (theFPGA.runFPGA()).str();
    theFPGA.write_ss_safe(ss.str());
}

int main(int argc, char** argv)
{

    int nevents = 5;
    std::string datadir = STRINGIFY(HLS4ML_DATA_DIR);
    std::string xclbinFilename = "";
    if (argc > 1) xclbinFilename = argv[1];
    if (argc > 2) nevents = atoi(argv[2]);
    if (argc > 3) datadir = argv[3];
    std::cout << "Will run " << nevents << " time(s), using " << datadir << " to get input features and output predictions (tb_input_features.dat and tb_output_predictions.dat)" << std::endl;

    size_t vector_size_in_bytes = sizeof(bigdata_t) * STREAMSIZE;
    size_t vector_size_out_bytes = sizeof(bigdata_t) * COMPSTREAMSIZE;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
    // is used if it is properly aligned. when not aligned, runtime had no choice but to create
    // its own host side buffer. So it is recommended to use this allocator if user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
    // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 
    //std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_in(STREAMSIZE*NUM_CU);
    //std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_hw_results(COMPSTREAMSIZE*NUM_CU);
    fpgaObj fpga;
    fpga.nevents = nevents;
    fpga.ikern = 0;
    fpga.source_in.reserve(STREAMSIZE*NUM_CU*NBUFFER);
    fpga.source_hw_results.reserve(COMPSTREAMSIZE*NUM_CU*NBUFFER);
    

    //initialize
    for(int j = 0 ; j < STREAMSIZE*NUM_CU*NBUFFER ; j++){
        fpga.source_in[j] = 0;
    }
    for(int j = 0 ; j < COMPSTREAMSIZE*NUM_CU*NBUFFER ; j++){
        fpga.source_hw_results[j] = 0;
    }

// OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    for (int i = 0; i < NUM_CU; i++) {
        cl::CommandQueue q_tmp(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        fpga.q.push_back(q_tmp);
    }
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    cl::Program::Binaries bins;
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    bins.push_back({buf,nb});

    devices.resize(1);
    cl::Program tmp_program(context, devices, bins);
    fpga.program = tmp_program;

    for (int ib = 0; ib < NBUFFER; ib++) {
        for (int i = 0; i < NUM_CU; i++) {
            std::string cu_id = std::to_string(i);
            std::string krnl_name_full =
                "alveo_hls4ml:{alveo_hls4ml_" + cu_id + "}";
            printf("Creating a kernel [%s] for CU(%d)\n",
                   krnl_name_full.c_str(),
                   i);
            //Here Kernel object is created by specifying kernel name along with compute unit.
            //For such case, this kernel object can only access the specific Compute unit
            cl::Kernel krnl_tmp = cl::Kernel(
                   fpga.program, krnl_name_full.c_str(), &fpga.err);
            fpga.krnl_xil.push_back(krnl_tmp);
        }
    }

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
    // Device-to-host communication
    
    fpga.writeList.reserve(NUM_CU*NBUFFER);
    fpga.kernList.reserve(NUM_CU*NBUFFER);
    fpga.readList.reserve(NUM_CU*NBUFFER);
    for (int ib = 0; ib < NBUFFER; ib++) {
        for (int ik = 0; ik < NUM_CU; ik++) {
            cl::Buffer buffer_in_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_bytes, fpga.source_in.data()+((ib*NUM_CU+ik) * STREAMSIZE));
            cl::Buffer buffer_out_tmp(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_size_out_bytes, fpga.source_hw_results.data()+((ib*NUM_CU+ik) * COMPSTREAMSIZE));
            fpga.buffer_in.push_back(buffer_in_tmp);
            fpga.buffer_out.push_back(buffer_out_tmp);
        
            cl::Event tmp_write = cl::Event();
            cl::Event tmp_kern = cl::Event();
            cl::Event tmp_read = cl::Event();
            fpga.write_event.push_back(tmp_write);
            fpga.kern_event.push_back(tmp_kern);
        
            int narg = 0;
            fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_in[ib*NUM_CU+ik]);
            fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_out[ib*NUM_CU+ik]);
            fpga.isFirstRun.push_back(true);
            std::vector<cl::Event> tmp_write_vec(1);
            std::vector<cl::Event> tmp_kern_vec(1);
            std::vector<cl::Event> tmp_read_vec(1);
            fpga.writeList.push_back(tmp_write_vec);
            fpga.kernList.push_back(tmp_kern_vec);
            fpga.readList.push_back(tmp_read_vec);
        }
    }

    auto t0 = Clock::now();
    auto t1 = Clock::now();
    auto t1a = Clock::now();
    auto t1b = Clock::now();
    auto t2 = Clock::now();
    auto t3 = Clock::now();

    for (int ib = 0; ib < NBUFFER; ib++) {
        for (int i = 0 ; i < NUM_CU ; i++){
            for (int istream = 0; istream < STREAMSIZE; istream++) {
            // Create the test data if no data files found or if end of files has been reached
      	        fpga.source_in[ib*NUM_CU*STREAMSIZE+i*STREAMSIZE+istream] = (bigdata_t)(12354.37674*(istream+STREAMSIZE*(ib+i+1)));
            }
        }
    }

    auto ts0 = SClock::now();
    print_nanoseconds("      begin:  ",ts0, 0);

    fpga.ithr = 0;
    std::thread th0(FPGA, std::ref(fpga));
    std::thread th1(FPGA, std::ref(fpga));
    std::thread th2(FPGA, std::ref(fpga));
    std::thread th3(FPGA, std::ref(fpga));
    std::thread th4(FPGA, std::ref(fpga));
    std::thread th5(FPGA, std::ref(fpga));
    std::thread th6(FPGA, std::ref(fpga));
    std::thread th7(FPGA, std::ref(fpga));
    th0.join();
    th1.join();
    th2.join();
    th3.join();
    th4.join();
    th5.join();
    th6.join();
    th7.join();
    auto ts4 = SClock::now();
    print_nanoseconds("       done:  ",ts4, 0);

    for (int i = 0 ; i < NUM_CU ; i++){
        OCL_CHECK(fpga.err, fpga.err = fpga.q[i].flush());
        OCL_CHECK(fpga.err, fpga.err = fpga.q[i].finish());
    }
// OPENCL HOST CODE AREA END
    auto ts5 = SClock::now();
    print_nanoseconds("       end:   ",ts5, 0);
    std::cout << fpga.ss.str();

    return EXIT_SUCCESS;
}

