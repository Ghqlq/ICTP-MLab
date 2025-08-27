############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
############################################################
open_project COO
set_top coo
add_files COO_spmv.h
add_files COO_spmv.cpp
add_files -tb COO_tb.cpp -cflags "-Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xczu3eg-sfvc784-1-e}
create_clock -period 10 -name default
config_export -format ip_catalog -output /home/student/Desktop/SpMV/COOordinate -rtl verilog
set_clock_uncertainty 1.25
source "./COO/solution1/directives.tcl"

