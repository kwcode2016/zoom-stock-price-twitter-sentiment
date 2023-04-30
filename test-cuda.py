import torch
import time

print ("NOTE!!: this works with python test-cuda.py  but not in conda base environment.")


# Create two large matrices (10000x10000) with random values
# a = torch.randn(10000, 10000)
a = torch.randn(20000, 20000)
# b = torch.randn(10000, 10000)
b = torch.randn(20000, 20000)

# Perform matrix multiplication on CPU
print ("CPU started: should take about 11 secs")
start_time_cpu = time.time()
result_cpu = torch.matmul(a, b)
elapsed_time_cpu = time.time() - start_time_cpu

# Move matrices to GPU (CUDA) if available
if torch.cuda.is_available():
    a = a.to('cuda')
    b = b.to('cuda')

    # Perform matrix multiplication on GPU (CUDA)
    print ("CUDA  started: should take about 2 secs! WOW!")
    start_time_cuda = time.time()
    result_cuda = torch.matmul(a, b)
    elapsed_time_cuda = time.time() - start_time_cuda

    print(f"CPU time: {elapsed_time_cpu:.2f} seconds")
    print(f"CUDA time: {elapsed_time_cuda:.2f} seconds")
else:
    print("CUDA is not available on your system.")
