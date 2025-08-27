############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_top -name spmv_ell "spmv_ell"
set_directive_pipeline "spmv_ell/spmv_ell_label0"
set_directive_array_partition -type complete -dim 2 "spmv_ell" ell_values
set_directive_array_partition -type complete -dim 2 "spmv_ell" ell_col_index
set_directive_array_partition -type complete -dim 1 "spmv_ell" x
set_directive_pipeline -II 2 "spmv_ell/L2"
