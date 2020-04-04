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
typedef std::chrono::high_resolution_clock Clock;

#include "xcl2.hpp"
#include <vector>
#include "kernel_params.h"

#define STRINGIFY2(var) #var
#define STRINGIFY(var) STRINGIFY2(var)

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
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_in(STREAMSIZE*4);
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_hw_results(COMPSTREAMSIZE*4);

    //initialize
    for(int j = 0 ; j < STREAMSIZE*4 ; j++){
        source_in[j] = 0;
    }
    for(int j = 0 ; j < COMPSTREAMSIZE*4 ; j++){
        source_hw_results[j] = 0;
    }

// OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
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
    cl::Program program(context, devices, bins);

    cl_int err;

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
    // Device-to-host communication
    std::vector<cl::Buffer> buffer_in(4);
    std::vector<cl::Buffer> buffer_out(4);
    for (int i = 0; i < 4; i++) {
        OCL_CHECK(err,
                  buffer_in[i] =
                      cl::Buffer(context,
                                 CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                 vector_size_in_bytes,
                                 source_in.data() + (i * STREAMSIZE),
                                 &err));
        OCL_CHECK(err,
                  buffer_out[i] =
                      cl::Buffer(context,
                                 CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                 vector_size_out_bytes,
                                 source_hw_results.data() + (i * COMPSTREAMSIZE),
                                 &err));
    }

    std::vector<cl::Kernel> krnls(4);
    for (int i = 0; i < 4; i++) {
        std::string cu_id = std::to_string(i);
        std::string krnl_name_full =
            "alveo_hls4ml:{alveo_hls4ml_" + cu_id + "}";
        printf("Creating a kernel [%s] for CU(%d)\n",
               krnl_name_full.c_str(),
               i);
        //Here Kernel object is created by specifying kernel name along with compute unit.
        //For such case, this kernel object can only access the specific Compute unit
        OCL_CHECK(err,
                  krnls[i] = cl::Kernel(
                      program, krnl_name_full.c_str(), &err));
    }

    for (int i = 0; i < 4; i++) {
        int narg = 0;

        //Setting kernel arguments
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_in[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_out[i]));
    }


    auto t0 = Clock::now();
    auto t1 = Clock::now();
    auto t1a = Clock::now();
    auto t1b = Clock::now();
    auto t2 = Clock::now();
    auto t3 = Clock::now();

    std::vector<cl::Event> write_event(4);
    std::vector<cl::Event> kern_event(4);
    std::vector<cl::Event> read_event(4);
    std::vector<std::vector<cl::Event>> waitList(4);
    std::vector<std::vector<cl::Event>> eventList(4);

    for (int i = 0 ; i < 4 ; i++){
        for (int istream = 0; istream < STREAMSIZE; istream++) {
        // Create the test data if no data files found or if end of files has been reached
  	    source_in[(i)*STREAMSIZE+istream] = (bigdata_t)(12354.37674*(istream+STREAMSIZE*(i+1)));
        }
    }
    for (int i = 0 ; i < nevents ; i++){
        t0 = Clock::now();
        std::vector<float> pr;
        int ikern = i%4;
        //for (int istream = 0; istream < COMPSTREAMSIZE; istream++) {
        //    source_hw_results[(ikern)*COMPSTREAMSIZE+istream/COMPRESSION] = 0;
        //}

        t1 = Clock::now();
        if (i >= 4) {
            OCL_CHECK(err, err = read_event[ikern].wait());
        }
    
        //Copy input data to device global memory
        OCL_CHECK(err,
                  err =
                      q.enqueueMigrateMemObjects({buffer_in[ikern]},
                                                 0 /* 0 means from host*/,
                                                 NULL,
                                                 &write_event[ikern]));

        t1a = Clock::now();
        waitList[ikern].clear();
        waitList[ikern].push_back(write_event[ikern]);
        //Launch the kernel
        OCL_CHECK(err,
                  err = q.enqueueNDRangeKernel(
                      krnls[ikern], 0, 1, 1, &waitList[ikern], &kern_event[ikern]));
        t1b = Clock::now();
        eventList[ikern].clear();
        eventList[ikern].push_back(kern_event[ikern]);
        OCL_CHECK(err,
                  err = q.enqueueMigrateMemObjects({buffer_out[ikern]},
                                                   CL_MIGRATE_MEM_OBJECT_HOST,
                                                   &eventList[ikern],
                                                   &read_event[ikern]));
    
        t2 = Clock::now();

        /*if (valid_data && !hit_end) {
            std::cout<<"Predictions: \n";
            for (int j = 0 ; j < STREAMSIZE ; j++){
                for (int k = 0 ; k < DATA_SIZE_OUT ; k++){
        	        std::cout << pr[j*DATA_SIZE_OUT + k] << " \t";
                }
            }
            std::cout << std::endl;
        }*/
        //std::cout<<"Quantized predictions: \n";
        //for (int j = 0 ; j < COMPSTREAMSIZE ; j++){
        //    std::cout << source_hw_results[(ikern)*COMPSTREAMSIZE+j] << " ";
        //}
        std::cout << std::endl;
        t3 = Clock::now();
        std::cout << " Prep time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << " ns" << std::endl;
        std::cout << " FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" << std::endl;
        std::cout << "    inputs: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1a - t1).count() << " ns" << std::endl;
        std::cout << "    kernel: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1b - t1a).count() << " ns" << std::endl;
        std::cout << "   outputs: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1b).count() << " ns" << std::endl;
        std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count() << " ns" << std::endl;
        std::cout<<"---- END EVENT "<<i+1<<" ----"<<std::endl;
    }
    OCL_CHECK(err, err = q.flush());
    OCL_CHECK(err, err = q.finish());
// OPENCL HOST CODE AREA END

    return EXIT_SUCCESS;
}
