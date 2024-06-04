"""
Specification of model parameters

"""

S_th = 1.5  # xB threshold
D_th = 0.5  # xB threshold

r_R1 = 0.5  # * 0.4
alpha1 = 20
r_01 = 1
r_Rt1 = 2  # * .4 #1
r_t1 = 3  # Expression rate with only t.f. bound.
k1 = 1

r_R2 = 0
r_02 = 1
r_Rt2 = 1
r_t2 = 2  # Expression rate with only t.f. bound.
k2 = 1

n = 4
a = .5**n

sigma = 0.01

params = [S_th, D_th, r_R1, alpha1, r_01, r_Rt1, 
          r_t1, k1, r_R2, r_02, r_Rt2, r_t2, k2, n, a, sigma]


"""
#These parameters give Huang's 3-attractor vector field
r_R1 = 0
r_01 = 1.
r_Rt1 = 1.
r_t1 = r_01 + r_Rt1 # Expression rate with only t.f. bound.
k1 = 1
nD = 10

r_R2 = 0
r_02 = 1.
r_Rt2 = 1.
r_t2 = r_02 + r_Rt2 # Expression rate with only t.f. bound.
k2 = 1

n = 4
a = .5**n
"""
