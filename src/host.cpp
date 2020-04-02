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

    int nevents = 1;
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
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_in(STREAMSIZE);
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_hw_results(COMPSTREAMSIZE);

    //initialize
    for(int j = 0 ; j < STREAMSIZE ; j++){
        source_in[j] = 0;
    }
    for(int j = 0 ; j < COMPSTREAMSIZE ; j++){
        source_hw_results[j] = 0;
    }

// OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
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

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
    // Device-to-host communication
    cl::Buffer buffer_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_in_bytes, source_in.data());
    cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            vector_size_out_bytes, source_hw_results.data());

    std::vector<cl::Memory> inBufVec, outBufVec;
    inBufVec.push_back(buffer_in);
    outBufVec.push_back(buffer_output);

    cl::Kernel krnl_alveo_hls4ml(program,"alveo_hls4ml");

    int narg = 0;
    krnl_alveo_hls4ml.setArg(narg++, buffer_in);
    krnl_alveo_hls4ml.setArg(narg++, buffer_output);

    //load input data from text file
    std::ifstream fin(datadir+"/tb_input_features.dat");
    //load predictions from text file
    std::ifstream fpr(datadir+"/tb_output_predictions.dat");
  
    std::string iline;
    std::string pline;
    int e = 0;
    bool hit_end = false;
    bool valid_data = true;
    if (!(fin.is_open()) || !(fpr.is_open())) {
        std::cout << "Unable to open input/predictions file, using random input" << std::endl;
        valid_data = false;
        //flag for success/failure of finding data files
    }
    std::ofstream fout;
    fout.open("tb_output_data.dat");

    auto t1 = Clock::now();
    auto t2 = Clock::now();

    for (int i = 0 ; i < nevents ; i++){
        std::vector<float> pr;
        for (int istream = 0; istream < STREAMSIZE; istream++) {
  	    if (valid_data && !hit_end){
            // If files are valid and their end has not been reached yet, get inputs/predictions from files
                if(std::getline(fin,iline) && std::getline(fpr,pline)) {
      	            if (e%1000==0) std::cout << "Processing event " << e << std::endl;
      	            e++;
                    char* cstr=const_cast<char*>(iline.c_str());
                    char* current;
                    std::vector<float> in;
                    current=strtok(cstr," ");
                    while(current!=NULL){
                        in.push_back(atof(current));
                        current=strtok(NULL," ");
                    }
                    cstr=const_cast<char*>(pline.c_str());
                    current=strtok(cstr," ");
                    while(current!=NULL){
                        pr.push_back(atof(current));
                        current=strtok(NULL," ");
                    }
                    for (int j = 0; j < DATA_SIZE_IN; j++) {
		      source_in[istream].range(16*(j+1)-1,16*j) =  ((data_t)in[j]).range(15,0);
                    }
		    //data_t test;
		    //test.range(15,0) = source_in[istream].range(16*18-1,16*17);
		    //std::cout << "input check ===> " << source_in[istream] << " -- " <<  source_in[istream].range(15,0) << " -- " << test << " -- " << ((data_t)in[10]) << std::endl;
		    if(istream % COMPRESSION == 0) source_hw_results[istream/COMPRESSION] = 0;
                    
                } else {
                    hit_end = true;
                }
            }
            else {
            // Create the test data if no data files found or if end of files has been reached
                for(int j = 0 ; j < DATA_SIZE_IN; j++){
  		  source_in[istream].range(16*(j+1)-1,16*j) = (data_t)(12.34*(j+DATA_SIZE_IN*STREAMSIZE*(i+1)));
                  //this is just a random number to produce dummy input data
                }
		if(istream % COMPRESSION == 0) source_hw_results[istream/COMPRESSION] = 0;
            }
        }
    
        t1 = Clock::now();
        // Copy input data to device global memory
        q.enqueueMigrateMemObjects(inBufVec,0/* 0 means from host*/);
        // Launch the Kernel
        // For HLS kernels global and local size is always (1,1,1). So, it is recommended
        // to always use enqueueTask() for invoking HLS kernel
        q.enqueueTask(krnl_alveo_hls4ml);
        // Copy Result from Device Global Memory to Host Local Memory
        q.enqueueMigrateMemObjects(outBufVec,CL_MIGRATE_MEM_OBJECT_HOST);
        // Check for any errors from the command queue
        q.finish();
        t2 = Clock::now();
        std::cout << "FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" << std::endl;

        if (valid_data && !hit_end) {
            std::cout<<"Predictions: \n";
            for (int j = 0 ; j < STREAMSIZE ; j++){
                for (int k = 0 ; k < DATA_SIZE_OUT ; k++){
        	        std::cout << pr[j*DATA_SIZE_OUT + k] << " \t";
                }
            }
            std::cout << std::endl;
        }
        /*std::cout<<"Quantized predictions: \n";
        for (int j = 0 ; j < COMPSTREAMSIZE ; j++){
            for (int k = 0 ; k < COMPRESSION ; k++){
	      data_t tmp;
	      tmp.range(15,0) = source_hw_results[j].range((k+1)*16-1,k*16);
	      std::cout << tmp  << " \n";
              fout << tmp  << " \n "; 
            }
            //fout << "\n";
        }*/
        std::cout << std::endl;
        std::cout<<"---- END EVENT "<<i+1<<" ----"<<std::endl;
    }
// OPENCL HOST CODE AREA END
    if (valid_data) {
        fin.close();
        fpr.close();
        fout.close();
    }

    return EXIT_SUCCESS;
}
