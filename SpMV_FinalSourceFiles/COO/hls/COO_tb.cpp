//
// Created by Ghala Buarish on 8/5/25.
//

#include "COO_spmv.h"
#include <stdio.h>
#include <fstream>
#include <iostream>

int main(){
	//coo_row_index = [0, 0, 1, 1, 2, 2, 2, 3, 3]
	//coo_col_index = [0, 1, 1, 2, 0, 2, 3, 1, 3]
	//coo_values = [3, 4, 5, 9, 2, 3, 1, 4, 6]

	hls::stream<DTYPE> y;
	hls::stream<DTYPE> input;

	DTYPE inputData[TOTAL] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 0, 1, 1, 2, 0, 2, 3, 1, 3, 3, 4, 5, 9, 2, 3, 1, 4, 6, 1, 2, 3, 4};
	for (int i=0; i<TOTAL; i++){
		input.write(inputData[i]);
	}

	coo(input, y);

	for(int i = 0; i < SIZE; i++){
		std::cout << y.read() << " ";
	}
}
