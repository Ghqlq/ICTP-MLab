//
// Created by Ghala Buarish on 8/5/25.
//

#include "COO_spmv.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>

void coo(hls::stream<DTYPE> &in_fifo, hls::stream<DTYPE> &out_fifo) {

#pragma HLS INTERFACE axis port = in_fifo
#pragma HLS INTERFACE axis port = out_fifo
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return

	int columnIndex[NNZ];
	int rowIndex[NNZ];
	DTYPE values[NNZ];
	DTYPE x[SIZE];
	DTYPE y[SIZE];

	for (int i=0; i< SIZE; i++){
		y[i] = 0.0;
	}
	read_rowInd:for (int i=0; i<NNZ; i++) {
		rowIndex[i] = (int)in_fifo.read();
	}
	read_colInd:for (int i = 0; i < NNZ; i++){

		columnIndex[i] = (int)in_fifo.read();
	}
	read_values:for (int i = 0; i < NNZ; i++) {
		values[i] = in_fifo.read();
	}
	read_input:for (int i = 0; i < SIZE; i++) {
		x[i] = in_fifo.read();
	}

	int currRow = rowIndex[0];

	coo_label0:for (int i = 0; i < NNZ; i++) {
		if (rowIndex[i] != currRow) { //if moved to next row, write y[prevRow]
			out_fifo.write(y[currRow]);
			currRow = rowIndex[i];
		}
		y[rowIndex[i]] += values[i] * x[columnIndex[i]];
	}
	//last row append
	out_fifo.write(y[currRow]);
}
