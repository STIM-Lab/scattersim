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

#root = "D:\\myGit\\build\\scatter_bld_winter\\"
#data_dir = "D:\\myGit\\build\\scatter_bld_winter\\lambdas\\"

scattervol_path = "D:\\myGit\\build\\scatter_bld_winter\\"
source_path = "C:\\Users\\sunrj\\Dropbox\\2023_Winter\\scattersim\\"      # Where the sample is from
data_path = scattervol_path + "movie_star\\"
sample_name = "star_40_x_y.npy"

files = os.listdir(data_path)
# for f in files:
#     file = os.path.join(data_path, f)
#     os.chmod(file, 0o777)
#     os.remove(file)

# #-----------X-Z SIM----------------------
# result_npy = scattervol_path + "xz.npy"
# save_axis = 1                           # Save xz plane
# N_lambda = 50
# lambda_min = 1
# lambda_max = 10
# # lambdas = np.linspace(lambda_max, lambda_min, N_lambda)
# nus = np.linspace(2 * np.pi / lambda_max, 2 * np.pi / lambda_min, N_lambda)
# lambdas = 1.0 / nus * 2 * np.pi
# # size = [28, 1, 10]                       # Size of the sample. 4 is the diamter of the cylinder(z)
# size = [20, 1, 4]                       # Size of the sample. 4 is the diamter of the cylinder(z)
# relative_slice = size[2]/2           # Use the bottom of the sample
# coefs = [80, 1]
# resolution = 8
# result_cw = scattervol_path + "volume.cw"

#-----------X-y SIM----------------------
result_npy = scattervol_path + "xy.npy"
save_axis = 2                           # Save xz plane
N_lambda = 50
lambda_min = 0.2
lambda_max = 3
# lambdas = np.linspace(lambda_max, lambda_min, N_lambda)
nus = np.linspace(2 * np.pi / lambda_max, 2 * np.pi / lambda_min, N_lambda)
lambdas = 1.0 / nus * 2 * np.pi
# size = [28, 1, 10]                       # Size of the sample. 4 is the diamter of the cylinder(z)
size = [10, 10, 0.1]                       # Size of the sample. 4 is the diamter of the cylinder(z)
relative_slice = size[2]/2           # Use the bottom of the sample
coefs = [40, 40]
resolution = 8
result_cw = scattervol_path + "volume.cw"

norm = plt.Normalize();

for lambdai in tqdm(range(len(lambdas))):
    if lambdai < 32:
        continue
    print("")
    print("----------"+ str(lambdai+1) + "th loop-------------")
    print("wavelength:" + str(lambdas[lambdai]))
    subprocess.run([scattervol_path+"scattervolume", "--sample", scattervol_path+sample_name, "--size", str(size[0]), str(size[1]), str(size[2]),
                    "--coef", str(coefs[0]), str(coefs[1]), "--lambda", str(lambdas[lambdai]), "--output", result_cw], shell=True, capture_output=False)

    # subprocess.run([scattervol_path+"scatterviewsample", "--input", result_cw, "--nogui", "--extent", str(size[0]), "--output",
    #                 result_npy, "--axis", str(save_axis), "--slice", str(2**(resolution-1)), "--resolution", str(resolution)], shell=True, capture_output=False)

    subprocess.run([scattervol_path+"scatterview", "--input", result_cw, "--size", str(size[0]),
                   "--nogui", "--resolution", str(resolution), "--output", result_npy,  "--axis", str(save_axis),
                    "--center", str(size[0]/2), str(size[0]/2), str(0), "--slice", str(relative_slice)], shell=True, capture_output=False)

    xz = np.load(result_npy)

    Ey = np.real(xz[:, :, save_axis])
    colors = plt.cm.RdYlBu_r(norm(Ey))[:, :, 0:3]
    ski.io.imsave(data_path+"Ey_" + str(lambdai).zfill(3) + ".jpg", colors)

    intensity = np.real(xz[:, :, 0] * np.conj(xz[:, :, 0]) + xz[:, :, 1] * np.conj(xz[:, :, 1]) + xz[:, :, 2] * np.conj(xz[:, :, 2]))
    colors = plt.cm.magma(norm(intensity))[:, :, 0:3]
    ski.io.imsave(data_path+"intensity_" + str(lambdai).zfill(3) + ".jpg", colors)

# subprocess.run("ffmpeg", "-i" "D:\\myGit\\build\\scatter_bld_winter\\lambdas\\*.png", "-o", 'video.mp4') \
#     .run())
stream = ffmpeg.input(data_path + "Ey_%03d.jpg", framerate=7)
stream = ffmpeg.output(stream, data_path + "Ey.mp4")
try:
    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True, overwrite_output=True)
except ffmpeg.Error as e:
    print('stdout:', e.stdout.decode('utf8'))
    print('stderr:', e.stderr.decode('utf8'))
    raise e
    
stream = ffmpeg.input(data_path + "intensity_%03d.jpg", framerate=7)
stream = ffmpeg.output(stream, data_path + "intensity.mp4")
try:
    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True, overwrite_output=True)
except ffmpeg.Error as e:
    print('stdout:', e.stdout.decode('utf8'))
    print('stderr:', e.stderr.decode('utf8'))
    raise e
#out, err = (ffmpeg.input(data_path + "*.jpg", pattern_type='glob', framerate=25).output(data_path+"video.mp4") \
#    .run())
#print(err)
a = 1