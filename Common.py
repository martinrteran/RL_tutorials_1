import torch
import time

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define tensor size — increase for more pressure
size = 8192  # 8K x 8K matrix
iterations = 500

# Preallocate tensors
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Warm-up
_ = torch.matmul(a, b)

# Benchmark loop
start = time.time()
for i in range(iterations):
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Ensure GPU finishes before timing
end = time.time()

print(f"Completed {iterations} multiplications of {size}x{size} matrices")
print(f"Total time: {end - start:.2f} seconds")
print(f"Result tensor is on device: {c.device}")