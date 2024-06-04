import numpy as np

array = np.array([0,1,-1,0,3])  # Your array here

# Find indices where array is zero
zero_indices = np.where(array == 0)
nonzero_indices = np.where(array != 0)

# Find indices where array is not zero by subtracting zero_indices from the range of indices
nonzero_indices2 = np.setdiff1d(np.arange(array.size), zero_indices)

print(zero_indices)
print(nonzero_indices)
print(nonzero_indices2)