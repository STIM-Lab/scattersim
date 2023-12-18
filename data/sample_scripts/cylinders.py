import numpy as np
import matplotlib.pyplot as plt

# size of the sample
S = 10

# number of samples and layers
N = 40
L = N

# radius of the cylinder
a = 1

# refractive index of the cylinder
n = 1.4 + 0.05j

x = np.linspace(-S/2.0, S/2.0, N)
X, Z = np.meshgrid(x, x)

R = np.sqrt(X**2 + Z**2)

C = np.ones_like(R, dtype=np.complex128)
C[R <= a] = n
C = np.expand_dims(C, 1)
C = C[int(N/2 - N/2*2*a/S):int(N/2 + N/2*2*a/S), :, :]

np.save("../cylinder_8_1_40_1.4_0.05j.npy", C)


