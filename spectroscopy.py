import numpy as np
import os
import subprocess
import glob
import skimage as ski
import skimage.io
import matplotlib.pyplot as plt
import matplotlib
import ffmpeg
from tqdm import tqdm
import shutil
from ScatterView import *
import math
import time

scattervol_path = "D:\\myGit\\build\\scatter_bld_winter\\"
source_path = "C:\\Users\\sunrj\\Dropbox\\2023_Winter\\scattersim\\"      # Where the sample is from
data_path = scattervol_path + "spectroscopy\\"
sample_name = "3.jpg"

# Configuration
lambdas = [12, 17, 32]
n = [1.111+0.199j, 1.376+0.429j, 1.546+0.324j]
size = [500, 500, 1]                       # Size of the sample. 4 is the diamter of the cylinder(z)
coefs = [40, 40]                          # Pixel number: 40*40
resolution = 8
save_axis = 2
relative_slice = size[0]/2+size[2]/2           # Use the bottom of the sample
sample_npy = scattervol_path + "3_npy.npy"
ref_npy = scattervol_path + "ref_npy.npy"
result_cw = scattervol_path + "result.cw"
result_npy = scattervol_path + "xy.npy"
recon_A = []

for lambdai in tqdm(range(len(lambdas))):
    start = time.time()
    print("")
    print("----------"+ str(lambdai+1) + "th loop-------------")
    print("wavelength:" + str(lambdas[lambdai]))
    print("n:" + str(n))
    # result_cw = scattervol_path + str(lambdas[lambdai]) + ".cw"

    # Create sample
    img = ski.io.imread(source_path + sample_name)[:, :, 0].astype(np.complex128) / 255
    img = (2 - img)
    img[img < 1.05] = 1 + 0j
    img[img > 1] = img[img > 1] / np.max(img) * n[lambdai]
    img[img < 1] = 1
    img = np.expand_dims(img, 0)
    np.save(sample_npy, img)  # Real sample
    img[:, :, :] = 1
    np.save(ref_npy, img)  # Reference

    # ----------------------------------Solve the reference first-------------------------------------
    subprocess.run([scattervol_path+"scattervolume", "--sample", ref_npy, "--size", str(size[0]), str(size[1]), str(size[2]),
                    "--coef", str(1), str(1), "--lambda", str(lambdas[lambdai]), "--output", result_cw], shell=True, capture_output=False)

    subprocess.run([scattervol_path+"scatterview", "--input", result_cw, "--size", str(size[0]),
                   "--nogui", "--resolution", str(resolution), "--output", result_npy,  "--axis", str(save_axis),
                    "--center", str(size[0]/2), str(size[0]/2), str(0), "--slice", str(relative_slice)], shell=True, capture_output=False)

    xy = np.load(result_npy)
    intensity_ref = np.real(
        xy[:, :, 0] * np.conj(xy[:, :, 0]) + xy[:, :, 1] * np.conj(xy[:, :, 1]) + xy[:, :, 2] * np.conj(xy[:, :, 2]))
    ref = time.time()
    print("---------------Profiling---------------")
    print("Reference intensity takes " + str(ref-start) +"s.")



    # ----------------------------Calculate the intensity for real sample------------------------------
    subprocess.run([scattervol_path+"scattervolume", "--sample", sample_npy, "--size", str(size[0]), str(size[1]), str(size[2]),
                    "--coef", str(coefs[0]), str(coefs[1]), "--lambda", str(lambdas[lambdai]), "--output", result_cw], shell=True, capture_output=False)
    sample_volume = time.time()
    print("Sample scattervolume takes " + str(sample_volume-ref) +"s.")
    subprocess.run([scattervol_path+"scatterview", "--input", result_cw, "--size", str(size[0]),
                   "--nogui", "--resolution", str(resolution), "--output", result_npy,  "--axis", str(save_axis),
                    "--center", str(size[0]/2), str(size[0]/2), str(0), "--slice", str(relative_slice)], shell=True, capture_output=False)

    xy = np.load(result_npy)
    intensity_sample = np.real(
        xy[:, :, 0] * np.conj(xy[:, :, 0]) + xy[:, :, 1] * np.conj(xy[:, :, 1]) + xy[:, :, 2] * np.conj(xy[:, :, 2]))
    sample_view = time.time()
    print("Sample scatterview takes " + str(sample_view-sample_volume) +"s.\n")

    A = np.log(intensity_ref[2**(resolution-1), 2**(resolution-1)] / intensity_sample[2**(resolution-1), 2**(resolution-1)])
    # Let's run scatterview instead. Might take less time
    # # Read the .cw file and evaluate the intensity on the lower boundary
    # layer = coupledwave()
    # layer.load(result_cw)
    
    recon_A.append(A)

    a = 1

print(recon_A)