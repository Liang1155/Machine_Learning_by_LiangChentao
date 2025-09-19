import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.style.use('seaborn-v0_8-whitegrid')

x = np.array([
    6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 8.3829, 7.4764, 8.5781, 6.4862, 5.0546,
    5.7107, 14.164, 5.734, 8.4084, 5.6407, 5.3794, 6.3654, 5.1301, 6.4296, 7.0708,
    6.1891, 20.27, 5.4901, 6.3261, 5.5649, 18.945, 12.828, 10.957, 13.176, 22.203,
    5.2524, 6.5894, 9.2482, 5.8918, 8.2111, 7.9334, 8.0959, 5.6063, 12.836, 6.3534,
    5.4069, 6.8825, 11.708, 5.7737, 7.8247, 7.0931, 5.0702, 5.8014, 11.7, 5.5416,
    7.5402, 5.3077, 7.4239, 7.6031, 6.3328, 6.3589, 6.2742, 5.6397, 9.3102, 9.4536,
    8.8254, 5.1793, 21.279, 14.908, 18.959, 7.2182, 8.2951, 10.236, 5.4994, 20.341,
    10.136, 7.3345, 6.0062, 7.2259, 5.0269, 6.5479, 7.5386, 5.0365, 10.274, 5.1077,
    5.7292, 5.1884, 6.3557, 9.7687, 6.5159, 8.5172, 9.1802, 6.002, 5.5204, 5.0594,
    5.7077, 7.6366, 5.8707, 5.3054, 8.2934, 13.394, 5.4369
], dtype=np.float32)

y = np.array([
    17.592, 9.1302, 13.662, 11.854, 6.8233, 11.886, 4.3483, 12, 6.5987, 3.8166,
    3.2522, 15.505, 3.1551, 7.2258, 0.71618, 3.5129, 5.3048, 0.56077, 3.6518, 5.3893,
    3.1386, 21.767, 4.263, 5.1875, 3.0825, 22.638, 13.501, 7.0467, 14.692, 24.147,
    -1.22, 5.9966, 12.134, 1.8495, 6.5426, 4.5623, 4.1164, 3.3928, 10.117, 5.4974,
    0.55657, 3.9115, 5.3854, 2.4406, 6.7318, 1.0463, 5.1337, 1.844, 8.0043, 1.0179,
    6.7504, 1.8396, 4.2885, 4.9981, 1.4233, -1.4211, 2.4756, 4.6042, 3.9624, 5.4141,
    5.1694, -0.74279, 17.929, 12.054, 17.054, 4.8852, 5.7442, 7.7754, 1.0173, 20.992,
    6.6799, 4.0259, 1.2784, 3.3411, -2.6807, 0.29678, 3.8845, 5.7014, 6.7526, 2.0576,
    0.47953, 0.20421, 0.67861, 7.5435, 5.3436, 4.2415, 6.7981, 0.92695, 0.152, 2.8214,
    1.8451, 4.2959, 7.2029, 1.9869, 0.14454, 9.0551, 0.61705
], dtype=np.float32)  

m = len(x)  # Number of samples


def compute_cost(w, b, x, y, m):
    h = w * x + b  # Linear model: h(x) = wx + b
    cost = (1/(2*m)) * np.sum((h - y)**2)  # MSE cost
    return cost




w_values = np.arange(-5, 10, 0.1)  

cost_values = [compute_cost(w, b=0, x=x, y=y, m=m) for w in w_values]


plt.figure(figsize=(10, 6))
plt.plot(w_values, cost_values, 'b-', linewidth=2)
plt.xlabel('Weight (w)', fontsize=12)
plt.ylabel('Cost (J(w))', fontsize=12)
plt.title('Cost Function vs Weight', fontsize=14)



plt.legend()
plt.show()

w_grid = np.linspace(-5, 10, 100)
b_grid = np.linspace(-10, 20, 100)
W, B = np.meshgrid(w_grid, b_grid)


J_values = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        J_values[i, j] = compute_cost(W[i,j], B[i,j], x, y, m)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, B, J_values, cmap='viridis', alpha=0.8)

ax.set_xlabel('Weight (w)', fontsize=12)
ax.set_ylabel('Bias (b)', fontsize=12)
ax.set_zlabel('Cost (J(w, b))', fontsize=12)
ax.set_title('3D Surface of Cost Function J(w, b)', fontsize=14)
fig.colorbar(surf, shrink=0.5, aspect=5)  
plt.show()
