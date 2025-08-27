//
// Created by Ghala Buarish on 7/17/25.
//

#include "CSR_baseline.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>

void spmv(hls::stream<DTYPE> &in_fifo, hls::stream<DTYPE> &out_fifo) {

#pragma HLS INTERFACE axis port = in_fifo
#pragma HLS INTERFACE axis port = out_fifo
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return


	int rowPtr[NUM_ROWS +1];
	int columnIndex[NNZ];
	DTYPE values[NNZ];
	DTYPE x[SIZE];
	DTYPE y[SIZE];


	spmv_label0:for (int i=0; i<NUM_ROWS+1; i++) {
		rowPtr[i] = (int)in_fifo.read();
	}

	spmv_label1:for (int i = 0; i < NNZ; i++){
		columnIndex[i] = (int)in_fifo.read();
	}
	spmv_label2:for (int i = 0; i < NNZ; i++) {
		values[i] = in_fifo.read();
	}

	spmv_label3:for (int i = 0; i < SIZE; i++) {
		x[i] = in_fifo.read();
	}


	L1: for (int i = 0; i < NUM_ROWS; i++) {

		DTYPE y0 = 0;
		L2: for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) {


			y0 += values[k] * x[columnIndex[k]];
		}
		out_fifo.write(y0);
	}

}

