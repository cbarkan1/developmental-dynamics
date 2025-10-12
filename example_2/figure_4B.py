import numpy as np
import matplotlib.pyplot as plt
from Example_2_Model import parameters

r0 = parameters['r0']
r = parameters['r']
K_sqrd = parameters['K_sqrd']
b = parameters['b']

def find_fixed_points_of_F(c1):
    roots = np.roots([-b, r0+r*c1, -b*K_sqrd, r0*K_sqrd])
    roots_real = [np.real(x) for x in roots if abs(np.imag(x))<1e-10]
    thresh = 15
    if len(roots_real)==3:
        return sorted(roots_real)[:]
    elif len(roots_real)==1:
        if roots_real[0]<thresh:
            return roots_real[0], np.nan, np.nan
        else:
            return np.nan, np.nan, roots_real[0]
    else:
        raise RuntimeError("real polynomial has odd number of complex roots?")


c1s_1 = np.linspace(0,7.1,100)
c1s_2 = np.linspace(7.222,9.5,100)
c1s_3 = np.linspace(9.6,16,100)
c1s = np.concatenate((c1s_1, c1s_2, c1s_3))
num_c1s = len(c1s)
left_roots = np.zeros(num_c1s)
middle_roots = np.zeros(num_c1s)
right_roots = np.zeros(num_c1s)
for i, c1 in enumerate(c1s):
    left_roots[i], middle_roots[i], right_roots[i] = find_fixed_points_of_F(c1)[:]

linewidth = 4
plt.figure(figsize=(4,2))
plt.plot(left_roots, c1s,'k', linewidth=linewidth)
plt.plot(middle_roots, c1s, color='red', linewidth=linewidth)
plt.plot(right_roots, c1s, 'k', linewidth=linewidth)
plt.xlim(0,70)
plt.ylim(5,11)
plt.show()
