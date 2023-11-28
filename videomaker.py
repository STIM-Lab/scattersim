import numpy as np
import os
import subprocess
import glob
import matplotlib.pyplot as plt
import ffmpeg

root = "D:\\myGit\\build\\scatter_bld_winter\\"
data_dir = "D:\\myGit\\build\\scatter_bld_winter\\lambdas\\"

lambdas = np.linspace(1, 5, 3)
size = [20, 1, 4]                       # Size of the sample. 4 is the diamter of the cylinder
coefs = [800, 1]
resolution = 6
save_axis = 1                           # Save xz plane
result_cw = root + "volume.cw"
result_npy = root + "xz.npy"

for lambdai in lambdas:
    subprocess.run([root+"scattervolume", "--sample", root+"cylinder.npy", "--size", str(size[0]), str(size[1]), str(size[2]),
                    "--coef", str(coefs[0]), str(coefs[1]), "--lambda", str(lambdai), "--output", result_cw], shell=True, capture_output=False)

    subprocess.run([root+"scatterviewsample", "--input", result_cw, "--nogui", "--extent", str(size[0]), "--output",
                    result_npy, "--axis", str(save_axis), "--slice", str(0), "--resolution", str(resolution)], shell=True, capture_output=False)

    xz = np.load(result_npy)

    Ey = xz[:, :, 1]
    plt.imshow(np.real(Ey))
    plt.set_cmap("bwr")
    plt.axis('off')
    plt.savefig(data_dir+"Ey_"+str(lambdai)+".jpg")

    intensity = xz[:, :, 0] ** 2 + xz[:, :, 1] ** 2 + xz[:, :, 2] ** 2
    plt.imshow(np.real(Ey))
    plt.set_cmap("magma")
    plt.axis('off')
    plt.savefig(data_dir+"intensity_"+str(lambdai)+".jpg")

# subprocess.run("ffmpeg", "-i" "D:\\myGit\\build\\scatter_bld_winter\\lambdas\\*.png", "-o", 'video.mp4') \
#     .run())
# stream = ffmpeg.input("D:\\myGit\\build\\scatter_bld_winter\\lambdas\\*.png")
# stream = ffmpeg.output(stream, "D:\\myGit\\build\\scatter_bld_winter\\lambdas\\video.mp4")
# ffmpeg.run(stream)
out, err = (ffmpeg.input(data_dir + "*.jpg", pattern_type='glob', framerate=25).output(data_dir+"video.mp4") \
    .run())

a = 1