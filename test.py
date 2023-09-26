import matplotlib.pyplot as plt
import numpy as np


# rand_arr = np.random.rand(10, 10)
# load arr
arr_path = "figures/singular_values_diff_lora.npy"
plt.figure(figsize=(8, 2))
# arr_path = "figures/singular_values_diff.npy"
# plt.figure(figsize=(8, 2))
arr = np.load(arr_path)[:, :45]
plt.imshow(arr, interpolation='nearest', aspect='auto', cmap='viridis')
plt.yticks(np.arange(len(arr)), labels=[])
plt.colorbar()
plt.xlabel('Singular Value Index')
plt.ylabel('Layer')
# plt.title('Cumulative Sum of Singular Values in Model Layers')
# plt.show()
# no white space
plt.savefig("figures/singular_values_diff_lora.pdf", bbox_inches='tight', dpi=300)