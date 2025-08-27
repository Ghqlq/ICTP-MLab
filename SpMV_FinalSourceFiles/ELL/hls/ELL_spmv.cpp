//
// Created by Ghala Buarish on 8/1/25.
//

#include "ELL_spmv.h"

#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

void spmv_ell(hls::stream<DTYPE> &in_fifo, hls::stream<DTYPE> &out_fifo) {

#pragma HLS INTERFACE axis port = in_fifo
#pragma HLS INTERFACE axis port = out_fifo
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return

	DTYPE ell_values[ROWS][MAX_NNZ];
	DTYPE ell_col_index[ROWS][MAX_NNZ];
	DTYPE x[ROWS];


	reading_values: for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < MAX_NNZ; j++) {
			ell_values[i][j] = in_fifo.read();
		}
	}
	reading_index: for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < MAX_NNZ; j++) {
			ell_col_index[i][j] = in_fifo.read();
		}
	}
	reading_x: for (int i = 0; i < ROWS; i++) {
		x[i] = in_fifo.read();
	}


	L1: for (int i = 0; i < ROWS; i++) {
		DTYPE y0 = 0;
		L2:for (int j = 0; j < MAX_NNZ; j++) {
			DTYPE col = ell_col_index[i][j];
			if (col != 4294967295) {
				y0 += ell_values[i][j] * x[col];
			}
		}
		out_fifo.write(y0);
	}

}



