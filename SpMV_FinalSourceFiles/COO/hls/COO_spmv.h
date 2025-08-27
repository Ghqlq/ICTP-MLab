//
// Created by Ghala Buarish on 8/5/25.
//

#ifndef COO_SPMV_H
#define COO_SPMV_H

#include <hls_stream.h>
#include <ap_axi_sdata.h>

const static int SIZE = 512;
const static int NNZ = 52429;
const static int TOTAL =  NNZ * 3 + SIZE;

typedef int DTYPE;

void coo(hls::stream<DTYPE> &in_fifo, hls::stream<DTYPE> &out_fifo);

#endif //COO_SPMV_H
