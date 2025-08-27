//
// Created by Ghala Buarish on 8/1/25.
//

#ifndef ELL_SPMV_H
#define ELL_SPMV_H

#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

const static int ROWS = 512;
const static int MAX_NNZ  = 103;
const static int TOTAL = ROWS * MAX_NNZ * 2 + ROWS; // ell_values + ell_col_index + x

typedef ap_int<32> DTYPE;

void spmv_ell(hls::stream<DTYPE> &in_fifo, hls::stream<DTYPE> &out_fifo);

#endif //ELL_SPMV_H
