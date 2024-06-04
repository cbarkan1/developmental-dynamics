import numpy as np
import time

# Generate a large array with random integers
array = np.random.randint(0, 2, size=1000000)

# Method 1: Calling np.where twice
start_time = time.time()
zero_indices_1, non_zero_indices_1 = np.where(array == 0), np.where(array != 0)
end_time = time.time()
method_1_time = end_time - start_time

# Method 2: Calling np.where once and using set operations
start_time = time.time()
zero_indices_2 = np.where(array == 0)
all_indices = np.arange(array.size)
non_zero_indices_2 = np.setdiff1d(all_indices, zero_indices_2)
end_time = time.time()
method_2_time = end_time - start_time

# Print the time taken by each method
print(f"Method 1 time: {method_1_time} seconds")
print(f"Method 2 time: {method_2_time} seconds")

# Verify that both methods give the same results
assert np.array_equal(zero_indices_1[0], zero_indices_2[0])
assert np.array_equal(non_zero_indices_1[0], non_zero_indices_2[0])
