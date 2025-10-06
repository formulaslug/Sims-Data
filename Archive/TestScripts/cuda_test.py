# import torch
from numba import cuda
import numpy as np
import timeit

@cuda.jit
def cudakernel1(array):
    thread_position = cuda.grid(1)
    array[thread_position] += 0.5

@cuda.jit
def cudakernel1b(array):
    thread_position = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    array[thread_position] += 0.5

@cuda.jit
def cuda_polyval(result, array, coeffs):
    # Evaluate a polynomial function over an array with Horner's method.
    # The coefficients are given in descending order.
    i = cuda.grid(1) # equivalent to i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    val = coeffs[0]
    for coeff in coeffs[1:]:
        val = val * array[i] + coeff
    result[i] = val

# array = np.zeros(1024 * 1024, np.float32)
# print('Initial array:', array)

# print('Kernel launch: cudakernel1[1024, 1024](array)')
# cudakernel1[1024, 1024](array)

# print('Updated array:', array)

# # Since it is a huge array, let's check that the result is correct:
# print('The result is correct:', np.all(array == np.zeros(1024 * 1024, np.float32) + 0.5))


array = np.random.randn(2048 * 1024 * 64).astype(np.float32)
coeffs = np.float32(range(1, 10))
result = np.empty_like(array)
cuda_polyval[2048, 1024](result, array, coeffs)
numpy_result = np.polyval(coeffs, array)
print('Maximum relative error compared to numpy.polyval:', np.max(np.abs(numpy_result - result) / np.abs(numpy_result)))



a = np.random.randn(2048 * 1024 * 64).astype(np.float32)  # our array
p = np.float32(range(1, 10))   # the coefficients of a polynomial in descending order
c = p[::-1] # the coefficients of the same polynomial in ascending order

# Measure execution time of np.polyval
time_polyval = timeit.timeit(lambda: np.polyval(p, a), number=10)
print(f"np.polyval execution time: {time_polyval:.6f} seconds")

# Measure execution time of np.polynomial.polynomial.polyval
time_polyval_alt = timeit.timeit(lambda: np.polynomial.polynomial.polyval(a, c), number=10)
print(f"np.polynomial.polynomial.polyval execution time: {time_polyval_alt:.6f} seconds")

# Measure execution time of np.polynomial.polynomial.polyval with tensor=False
time_polyval_tensor = timeit.timeit(lambda: np.polynomial.polynomial.polyval(a, c, tensor=False), number=10)
print(f"np.polynomial.polynomial.polyval (tensor=False) execution time: {time_polyval_tensor:.6f} seconds")

# Measure execution time of cuda_polyval
time_cuda_polyval = timeit.timeit(lambda: cuda_polyval[2048, 1024](result, array, coeffs), number=10)
print(f"cuda_polyval execution time: {time_cuda_polyval:.6f} seconds")