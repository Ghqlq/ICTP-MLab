//
// Created by Ghala Buarish on 7/17/25.
//

#ifndef CSR_BASELINE_H
#define CSR_BASELINE_H

#include <hls_stream.h>
#include <ap_axi_sdata.h>

const static int SIZE = 4;
const static int NNZ = 3;
const static int NUM_ROWS = SIZE;
const static int TOTAL = NUM_ROWS+1 + NNZ + NNZ + SIZE;

typedef int DTYPE;

void spmv(hls::stream<DTYPE> &in_fifo, hls::stream<DTYPE> &out_fifo);

#endif // CSR_BASELINE_H
