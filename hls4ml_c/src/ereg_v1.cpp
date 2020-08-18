//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "ereg_v1.h"

//hls-fpga-machine-learning insert weights
#include "weights/s2.h"
#include "weights/b2.h"
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/s5.h"
#include "weights/b5.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/s8.h"
#include "weights/b8.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w11.h"
#include "weights/b11.h"

void ereg_v1(
    input_t input[N_INPUT_1_1],
    layer11_t layer11_out[N_LAYER_11]
//    unsigned short &const_size_in_1,
//    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    //#pragma HLS ARRAY_RESHAPE variable=input complete dim=0 
    //#pragma HLS ARRAY_RESHAPE variable=layer11_out complete dim=0 
    #pragma HLS INTERFACE ap_vld port=input,layer11_out 
    #pragma HLS PIPELINE 


#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 18>(s2, "s2.txt");
        nnet::load_weights_from_txt<model_default_t, 18>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 558>(w3, "w3.txt");
        nnet::load_weights_from_txt<model_default_t, 31>(b3, "b3.txt");
        nnet::load_weights_from_txt<model_default_t, 31>(s5, "s5.txt");
        nnet::load_weights_from_txt<model_default_t, 31>(b5, "b5.txt");
        nnet::load_weights_from_txt<model_default_t, 341>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 11>(b6, "b6.txt");
        nnet::load_weights_from_txt<model_default_t, 11>(s8, "s8.txt");
        nnet::load_weights_from_txt<model_default_t, 11>(b8, "b8.txt");
        nnet::load_weights_from_txt<model_default_t, 33>(w9, "w9.txt");
        nnet::load_weights_from_txt<model_default_t, 3>(b9, "b9.txt");
        nnet::load_weights_from_txt<model_default_t, 3>(w11, "w11.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b11, "b11.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(input, layer2_out, s2, b2);

    layer3_t layer3_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::dense_latency<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3);

    layer4_t layer4_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::relu<layer3_t, layer4_t, relu_config4>(layer3_out, layer4_out);

    layer5_t layer5_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::normalize<layer4_t, layer5_t, config5>(layer4_out, layer5_out, s5, b5);

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense_latency<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6);

    layer7_t layer7_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out);

    layer8_t layer8_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::normalize<layer7_t, layer8_t, config8>(layer7_out, layer8_out, s8, b8);

    layer9_t layer9_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::dense_latency<layer8_t, layer9_t, config9>(layer8_out, layer9_out, w9, b9);

    layer10_t layer10_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::relu<layer9_t, layer10_t, relu_config10>(layer9_out, layer10_out);

    nnet::dense_latency<layer10_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11);

}
