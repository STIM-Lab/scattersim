# This script creates a numpy file that represents a layered homogeneous sample
# for input into scattervol. The default should match the default input of scatterlayer.

import numpy as np

V = np.array((complex(1.4, 0)))
V = np.reshape(V, (1, 1, 1))
np.save("layers.npy", V)