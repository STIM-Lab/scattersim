import numpy as np
import scipy as sp
import skimage.io
import matplotlib.pyplot as plt

eta = complex(1.4, 0.01)
infile = "triangle.png"
outfile = "triangle.npy"

I = skimage.io.imread(infile)[:, :, 0] / 255

SAMPLE = I * (eta - 1.0) + 1.0
# SAMPLE = np.asfortranarray(SAMPLE)
np.save(outfile, SAMPLE)

#%% display the simulated image from a specified number of Fourier coefficients
M = 300

I_fft_shift = sp.fft.fft2(I)
I_fft = sp.fft.fftshift(I_fft_shift)

center = np.array([ int(I.shape[0]/2), int(I.shape[1]/2)])
x0 = center[0] - int(M/2)
x1 = x0 + M

y0 = center[1] - int(M/2)
y1 = y0 + M

I_fft_cropped = I_fft[x0:x1, y0:y1]
I_fft_cropped_shift = sp.fft.ifftshift(I_fft_cropped)

I_cropped = np.real(sp.fft.ifft2(I_fft_cropped_shift))

plt.imshow(I_cropped)
plt.show()