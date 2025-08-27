import time
import tracemalloc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

'''Code to test 5 algorithms with different compression formats to perform sparse matrix vector multiplication.
    algorithms are tested using 4x4 and 10x10 matrices.
    '''

#  input vector
x = [1, 2, 3, 4]
x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

''' 

1) Compressed Sparse Row (CSR)

'''
def spmv_csr(row_ptr, col_index, values, x):
    num_rows = len(row_ptr) - 1
    y = [0.0] * num_rows
    for i in range(num_rows):
        sum_val = 0
        for k in range(row_ptr[i], row_ptr[i + 1]):
            sum_val += values[k] * x[col_index[k]]
        y[i] = sum_val
    return y

# Testing CSR (4x4):
csr_row_ptr = [0, 2, 4, 7, 9]
csr_col_index = [0, 1, 1, 2, 0, 2, 3, 1, 3]
csr_values = [3, 4, 5, 9, 2, 3, 1, 4, 6]

# Testing CSR (10x10):
csr_row_ptr2 = [0, 4, 6, 7, 10, 11, 16, 19, 24, 27, 30]
csr_col_index2 = [1, 2, 7, 9, 1, 2, 5, 3, 4, 5, 3, 0, 1, 2, 3, 5, 2, 7, 9, 0, 3, 4, 5, 6, 0, 6, 8, 1, 2, 4]
csr_values2 = [5, 1, 3, 1, 5, 3, 2, 5, 5, 2, 4, 5, 2, 4, 4, 4, 3, 2, 5, 2, 2, 1, 1, 2, 3, 1, 4, 1, 2, 2]

''' 

2) Compressed Sparse Column (CSC)

'''

def spmv_csc(col_ptr, row_index, values, x):
    num_rows = max(row_index) + 1
    y = [0.0] * num_rows
    for j in range(len(col_ptr) - 1):
        for k in range(col_ptr[j], col_ptr[j + 1]):
            y[row_index[k]] += values[k] * x[j]
    return y

# CSC data (4x4):
csc_col_ptr = [0, 2, 5, 7, 9]
csc_row_index = [0, 2, 0, 1, 3, 1, 2, 2, 3]
csc_values = [3, 2, 4, 5, 4, 9, 3, 1, 6]

# CSC data (10x10):
csc_col_ptr2 = [0, 3, 7, 12, 16, 19, 23, 25, 27, 28, 30]
csc_row_index2 = [5, 7, 8, 0, 1, 5, 9, 0, 1, 5, 6, 9, 3, 4, 5, 7, 3, 7, 9, 2, 3, 5, 7, 7, 8, 0, 6, 8, 0, 6]
csc_values2 = [5, 2, 3, 5, 5, 2, 1, 1, 3, 4, 3, 2, 5, 4, 4, 2, 5, 1, 2, 2, 2, 4, 1, 2, 1, 3, 2, 4, 1, 5]


''' 

3) Coordinate (COO) Formate

'''

def spmv_coo(row_index, col_index, values, x):
    num_rows = max(row_index) + 1
    y = [0.0] * num_rows
    for i in range(len(values)):
        y[row_index[i]] += values[i] * x[col_index[i]]
    return y

# COO data (4x4):
coo_row_index = [0, 0, 1, 1, 2, 2, 2, 3, 3]
coo_col_index = [0, 1, 1, 2, 0, 2, 3, 1, 3]
coo_values = [3, 4, 5, 9, 2, 3, 1, 4, 6]

# COO data (10x10):
coo_row_index2 = [0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9]
coo_col_index2 = [1, 2, 7, 9, 1, 2, 5, 3, 4, 5, 3, 0, 1, 2, 3, 5, 2, 7, 9, 0, 3, 4, 5, 6, 0, 6, 8, 1, 2, 4]
coo_values2 = [5, 1, 3, 1, 5, 3, 2, 5, 5, 2, 4, 5, 2, 4, 4, 4, 3, 2, 5, 2, 2, 1, 1, 2, 3, 1, 4, 1, 2, 2]


''' 

4) ELLPACK (ELL) Format

'''

def spmv_ell(ell_values, ell_col_index, x):
    num_rows = len(ell_values)
    max_nnz = len(ell_values[0])
    y = [0.0] * num_rows
    for i in range(num_rows):
        for j in range(max_nnz):
            y[i] += ell_values[i][j] * x[ell_col_index[i][j]]
    return y

# ELL data (4x4):
ell_values = [
    [3, 4, 0],
    [5, 9, 0],
    [2, 3, 1],
    [4, 6, 0],
]
ell_col_index = [
    [0, 1, -1],
    [1, 2, -1],
    [0, 2, 3],
    [1, 3, -1],
]

# ELL data (10x10):
ell_values2 = [
    [5, 1, 3, 1, 0],
    [5, 3, 0, 0, 0],
    [2, 0, 0, 0, 0],
    [5, 5, 2, 0, 0],
    [4, 0, 0, 0, 0],
    [5, 2, 4, 4, 4],
    [3, 2, 5, 0, 0],
    [2, 2, 1, 1, 2],
    [3, 1, 4, 0, 0],
    [1, 2, 2, 0, 0]
]

ell_col_index2 = [
    [1, 2, 7, 9, -1],
    [1, 2, -1, -1, -1],
    [5, -1, -1, -1, -1],
    [3, 4, 5, -1, -1],
    [3, -1, -1, -1, -1],
    [0, 1, 2, 3, 5],
    [2, 7, 9, -1, -1],
    [0, 3, 4, 5, 6],
    [0, 6, 8, -1, -1],
    [1, 2, 4, -1, -1]
]


''' 

5) Sliced ELLPACK (SELL)

'''

def spmv_sell(slice_height, slice_offsets, values, col_index, x):
    y = []
    offset = 0
    num_slices = len(slice_offsets)
    for slice_num in range(num_slices):
        rows_in_slice = min(slice_height, len(x) - slice_num * slice_height)
        cols_in_slice = (slice_offsets[slice_num] - offset) // rows_in_slice
        for i in range(rows_in_slice):
            sum_val = 0
            for j in range(cols_in_slice):
                idx = offset + j * rows_in_slice + i
                col = col_index[idx]
                val = values[idx]
                if col != -1:
                    sum_val += val * x[col]
            y.append(sum_val)
        offset = slice_offsets[slice_num]
    return y

# SELL data (4x4):
slice_height = 2

sell_values = [3, 5, 4, 9, 0, 0, 0, 0,
    2, 4, 0, 0, 3, 0, 1, 6]

sell_col_index = [
    0, 1, 1, 2, -1, -1, -1, -1,
    0, 1, -1, -1, 2, -1, 3, 3]

slice_offsets = [8, 16]

# SELL data (10x10):
sell_values2 = [5, 5, 1, 3, 3, 0, 1, 0, 2, 5, 0, 5, 0, 2, 4, 5, 0, 2, 0, 4, 0, 4, 0, 4, 3,
                2, 2, 2, 5, 1, 0, 1, 0, 2, 3, 1, 1, 2, 4, 2]
sell_col_index2 = [1, 1, 2, 2, 7, -1, 9, -1, 5, 3, -1, 4, -1, 5, 3, 0, -1, 1, -1, 2, -1, 3,
                   -1, 5, 2, 0, 7, 3, 9, 4, -1, 5, -1, 6, 0, 1, 6, 2, 8, 4]
slice_offsets2 = [8, 14, 24, 34, 40]


'''
    ----------------------------------------------------
    ----------------------------------------------------
    ----------------------------------------------------
    
'''
performance_data = []

def measure_performance(name, input_size):
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            exec_time = end - start
            peak_kb = peak / 1024
            print(f"{name} | Input: {input_size} | Time: {exec_time:.6f}s | Memory: {peak_kb:.2f} KB")
            performance_data.append({
                "Algorithm": name,
                "Input": input_size,
                "Time (s)": exec_time,
                "Memory (KB)": peak_kb
            })
            return result
        return wrapper
    return decorator


@measure_performance("CSR", len(x))
def run_csr():
    return spmv_csr(csr_row_ptr, csr_col_index, csr_values, x)

@measure_performance("CSC", len(x))
def run_csc():
    return spmv_csc(csc_col_ptr, csc_row_index, csr_values, x)

@measure_performance("COO", len(x))
def run_coo():
    return spmv_coo(coo_row_index, coo_col_index, coo_values, x)

@measure_performance("ELL", len(x))
def run_ell():
    return spmv_ell(ell_values, ell_col_index, x)

@measure_performance("SELL", len(x))
def run_sell():
    return spmv_sell(slice_height, slice_offsets, sell_values, sell_col_index, x)

run_csr()
run_csc()
run_coo()
run_ell()
run_sell()


''' 10x10 MATRIX'''

@measure_performance("CSR", len(x2))
def run_csr2():
    return spmv_csr(csr_row_ptr2, csr_col_index2, csr_values2, x2)

@measure_performance("CSC", len(x2))
def run_csc2():
    return spmv_csc(csc_col_ptr2, csc_row_index2, csr_values2, x2)

@measure_performance("COO", len(x2))
def run_coo2():
    return spmv_coo(coo_row_index2, coo_col_index2, coo_values2, x2)

@measure_performance("ELL", len(x2))
def run_ell2():
    return spmv_ell(ell_values2, ell_col_index2, x2)

@measure_performance("SELL", len(x2))
def run_sell2():
    return spmv_sell(slice_height, slice_offsets2, sell_values2, sell_col_index2, x2)

run_csr2()
run_csc2()
run_coo2()
run_ell2()
run_sell2()


df = pd.DataFrame(performance_data)
df_4x4 = df[df["Input"] == 4]
df_10x10 = df[df["Input"] == 10]

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
for alg in df_4x4["Algorithm"].unique():
    alg_df = df_4x4[df_4x4["Algorithm"] == alg]
    plt.plot(alg_df["Input"], alg_df["Time (s)"], marker='o', label=alg)
plt.title("4x4 Execution Time")
plt.xlabel("Input Size (n)")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
for alg in df_4x4["Algorithm"].unique():
    alg_df = df_4x4[df_4x4["Algorithm"] == alg]
    plt.plot(alg_df["Input"], alg_df["Memory (KB)"], marker='o', label=alg)
plt.title("4x4 Memory Usage")
plt.xlabel("Input Size (n)")
plt.ylabel("Memory (KB)")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
for alg in df_10x10["Algorithm"].unique():
    alg_df = df_10x10[df_10x10["Algorithm"] == alg]
    plt.plot(alg_df["Input"], alg_df["Time (s)"], marker='o', label=alg)
plt.title("10x10 Execution Time")
plt.xlabel("Input Size (n)")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
for alg in df_10x10["Algorithm"].unique():
    alg_df = df_10x10[df_10x10["Algorithm"] == alg]
    plt.plot(alg_df["Input"], alg_df["Memory (KB)"], marker='o', label=alg)
plt.title("10x10 Memory Usage")
plt.xlabel("Input Size (n)")
plt.ylabel("Memory (KB)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
