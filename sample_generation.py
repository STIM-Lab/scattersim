import numpy as np

pos_interest = [-100, 100, -100, 100, -100, 100]
num_pixels = [602, 2, 100]     # shape would be (8, 4, 50)
n_layers = [1, 1]
n_medium = 1.0
n_sample = 1.4+0.07j
center = [0, 0, 0]
width = 50
z = [-3.5, 3.5]
# def Grating():
    # X = np.linspace(pos_interest[0], pos_interest[1], num_pixels[0])
    # Y = np.linspace(pos_interest[2], pos_interest[3], num_pixels[1])
    # Z = np.linspace(pos_interest[4], pos_interest[5], num_pixels[2])
    # sample = np.meshgrid(Y, Z, X)
    # index = np.ones(np.array(sample[0]).shape, dtype='complex128') * n_medium
    # fu = np.zeros(num_pixels[0])
    # fu[num_pixels[0] // 2] = 600
    # fu[int(num_pixels[0] / 2 - frq)] = -100
    # fu[int(num_pixels[0] / 2 + frq)] = -100
    # fu_shift = np.fft.ifftshift(fu)
    # fx = np.fft.ifft(fu_shift)
    # fx = np.fft.fftshift(fx)
    # # index[:,:,:] = fx
    # index[int((center[2] - height // 2 - pos_interest[4]) / (pos_interest[5] - pos_interest[4]) * num_pixels[2]): int((center[2] + height // 2 - pos_interest[4]) / (pos_interest[5] - pos_interest[4]) * num_pixels[2]),
    # :, :] = fx
    # 
    # Z = Z
    # return index

# 
# def Sphere():
#     # Refer to: https://blog.csdn.net/yuzeyuan12/article/details/108572868
#     X = np.linspace(pos_interest[0], pos_interest[1], num_pixels[0]).reshape([1, 1, num_pixels[0]])
#     Y = np.linspace(pos_interest[2], pos_interest[3], num_pixels[1]).reshape([1, num_pixels[1], 1])
#     Z = np.linspace(pos_interest[4], pos_interest[5], num_pixels[2]).reshape([num_pixels[2], 1, 1])
#     sample = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2 <= radius**2
#     index = np.ones(sample.shape, dtype='complex128') * n_medium
#     index[sample] = n_sample
#     Z = Z
#     return index

def RecBar():
    # one-layer rectangular bar sample for test only.
    height = z[-1] - z[0]
    X = np.linspace(pos_interest[0], pos_interest[1], num_pixels[0])
    Y = np.linspace(pos_interest[2], pos_interest[3], num_pixels[1])
    Z = np.linspace(pos_interest[4], pos_interest[5], num_pixels[2])
    sample = np.meshgrid(X, Y, Z)
    index = np.ones(np.array(sample[0]).shape, dtype='complex128') * n_medium
    index = np.ones(np.array(sample[0]).shape, dtype='complex128') * n_layers[0]
    index[:, :, Z > z[-1]] = n_layers[-1]
    # index[int((center[2]-height/2 - pos_interest[4])/(pos_interest[5]-pos_interest[4]) * num_pixels[2]): int((center[2]+height/2 - pos_interest[4])/(pos_interest[5]-pos_interest[4]) * num_pixels[2]), :,
    # int((center[0]-width/2 - pos_interest[0])/(pos_interest[1]-pos_interest[0]) * num_pixels[0]) : int((center[0]+width/2 - pos_interest[0])/(pos_interest[1]-pos_interest[0]) * num_pixels[0])] = n_sample
    index[:, int((center[0]-width/2 - pos_interest[0])/(pos_interest[1]-pos_interest[0]) * num_pixels[0]) : int((center[0]+width/2 - pos_interest[0])/(pos_interest[1]-pos_interest[0]) * num_pixels[0]), int((center[2]-height/2 - pos_interest[4])/(pos_interest[5]-pos_interest[4]) * num_pixels[2]): int((center[2]+height/2 - pos_interest[4])/(pos_interest[5]-pos_interest[4]) * num_pixels[2])] = n_sample

    return index

sample = RecBar()
sample = np.asfortranarray(sample)
import matplotlib.pyplot as plt
plt.imshow(np.real(sample[0, :, :]))
plt.show()

outFile = "D:/myGit/build\coupledwave/RecBar.npy"
np.save(outFile, sample)
