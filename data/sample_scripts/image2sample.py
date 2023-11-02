import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

def resample_layers(image, rate):
    
    if rate == 0:
        return image
    return image[::rate, :]
    
    


I = ski.io.imread("cougar.png")[:, :, 0].astype(np.float32) / 255

width = I.shape[1]
height = I.shape[0]

I_opaque = (1 - I) * 1.4

I = I + I_opaque


Lsamples = 5
UVsamples = 5
Ldrop = 3
UVdrop = 3


layer_rate = 1

for lsi in range(Lsamples):
    
        
        Isub = resample_layers(I, layer_rate)
        
        uv_rate = 1
        for uvsi in range(UVsamples):
            
            if uv_rate == 0:
                uv_rate = 1
            coefficients = int(Isub.shape[1] / uv_rate)
            half_coefficients = int(coefficients/2)
            
            
            FFTsub = np.fft.fft(Isub, axis=1)
            
            a = FFTsub[:, 0:half_coefficients]
            b = FFTsub[:, -half_coefficients:]
            FFTcropped = np.concatenate((a, b), axis=1)
            
            Icropped = np.fft.ifft(FFTcropped, axis=1)
        
            plt.subplot(Lsamples, UVsamples, uvsi * UVsamples + lsi + 1)
            plt.imshow(np.abs(Icropped), extent=(0, width, 0, height), interpolation="nearest")
            plt.title("Layers: " + str(Icropped.shape[0]) + ", Coefficients: " + str(coefficients))
            
            uv_rate *= UVdrop
            
        layer_rate *= Ldrop

