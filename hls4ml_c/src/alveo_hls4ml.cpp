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

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce latency and 
    device resource utilization of the resulting RTL code
    This is a wrapper to be used with an hls4ml project to enable proper handling by SDAccel
*******************************************************************************/

#define PROJ_HDR <MYPROJ.h>

#include PROJ_HDR
#include "kernel_params.h"

/*
    HLS4ML Kernel Implementation 
    Arguments:
        in    (input)     --> Input Vector
        out   (output)    --> Output Vector
   */
extern "C" {
void alveo_hls4ml(
        const bigdata_t *in, // Read-Only Vector
        bigdata_t *out       // Output Result
        )
{
// SDAccel kernel must have one and only one s_axilite interface which will be used by host application to configure the kernel.
// Here bundle control is defined which is s_axilite interface and associated with all the arguments (in and out),
// control interface must also be associated with "return".
// All the global memory access arguments must be associated to one m_axi(AXI Master Interface). Here all two arguments(in, out) are 
// associated to bundle gmem which means that a AXI master interface named "gmem" will be created in Kernel and all these variables will be 
// accessing global memory through this interface.
// Multiple interfaces can also be created based on the requirements. For example when multiple memory accessing arguments need access to
// global memory simultaneously, user can create multiple master interfaces and can connect to different arguments.
#pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=in   bundle=control
#pragma HLS INTERFACE s_axilite port=out  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS DATAFLOW
    //necessary for hls4ml kernel, not used

    bigdata_t in_bigbuf[STREAMSIZE];
    bigdata_t out_bigbuf[COMPSTREAMSIZE];
    
    input_t in_buf[STREAMSIZE][DATA_SIZE_IN];
    layer11_t out_buf[STREAMSIZE][DATA_SIZE_OUT];
    //these will get partitioned properly in the hls4ml code

    #pragma HLS ARRAY_RESHAPE   variable=in_buf  complete dim=2
    #pragma HLS ARRAY_RESHAPE   variable=out_buf complete dim=2

    //getting data from DDR
    for (int i = 0; i < STREAMSIZE; i++) {
      in_bigbuf[i] = in[i];
    }
    for (int i = 0; i < STREAMSIZE; i++) {
      #pragma HLS PIPELINE
      for(int i0 = 0; i0 < DATA_SIZE_IN; i0++) { 
         #pragma HLS UNROLL
	 in_buf[i][i0].range(15,0) = in_bigbuf[i].range(16*(i0+1)-1,16*i0);
      }
    }
    //run inference
    for (int i = 0; i < STREAMSIZE; i++) {
      #pragma HLS DATAFLOW
      hls4ml: MYPROJ(in_buf[i],out_buf[i]);
    }
    for (int i = 0; i < COMPSTREAMSIZE; i++) {
      #pragma HLS PIPELINE
      bigdata_t tmp;
      for(int i1 = 0; i1 < COMPRESSION;i1++) { 
       for(int i0 = 0; i0 < DATA_SIZE_OUT; i0++) { 
        #pragma HLS UNROLL
	tmp((i1+1)*16-1,(i1)*16) = out_buf[i*COMPRESSION+i1][i0].range(15,0);
       }
      }
      out_bigbuf[i] = tmp;
    }
    //place output into DDR
    for (int i = 0; i < COMPSTREAMSIZE; i++) {
     out[i] = out_bigbuf[i];
    }
  }
}

