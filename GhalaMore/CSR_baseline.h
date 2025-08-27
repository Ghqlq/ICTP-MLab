//
// Created by Ghala Buarish on 7/17/25.
//

#ifndef CSR_BASELINE_H
#define CSR_BASELINE_H


const static int SIZE = 4;       // Size of square matrix (4x4)
const static int NNZ = 9;        // Number of non-zero elements
const static int NUM_ROWS = 4;   // Number of rows (same as SIZE)

typedef float DTYPE;

void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
		  DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE]);

#endif CSR_BASELINE_H

