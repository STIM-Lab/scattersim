import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import glob
import time
import math
from tqdm import tqdm

def cart2sph(x, y, z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r

# Calculates the intensity of a complex X, Y, Z, 3 vector field
def intensity(E):
    Econj = np.conj(E)                                                          # calculate the complex conjugate
    I = np.sum(E * Econj, axis=-1)
    return np.real(I)

# generate a set of points representing a rectangular focal spot
# size is a tuple giving the span of the focal spot
# samples is a tuple giving the number of points used to build the focal spot
# theta is the rotation angle about the z-axis [0, 2pi]
# phi is the rotation angle about the y-axis [0, pi/2]
def generate_points(span, samples, phi):
    rotate = True
    x = np.linspace(-span[0]/2, span[0]/2, samples[0])
    y = np.linspace(-span[1]/2, span[1]/2, samples[1])
    z = np.linspace(-span[2]/2, span[2]/2, samples[2])
    
    X, Y, Z = np.meshgrid(x, y, z)
    
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    
    Xn = cos_phi * X + sin_phi * Z
    Yn = Y
    Zn = -sin_phi * X + cos_phi * Z

    return (np.ndarray.flatten(Xn), np.ndarray.flatten(Yn), np.ndarray.flatten(Zn))

# simulates a single focused field and saves the resulting cw file
# filename - specify the name of the cw file that will be saved
# position - 3D position of the focal point near the samples
# direction - incoming illumination direction
# wavelength - wavelength of the field
# num_waves - number of samples used to simulate the field
# alpha_aperture - outer aperture angle of the objective
# beta_obscuration - angle of the center obscuration
def simulate_point(filename, position, direction, wavelength, num_waves, alpha_aperture, beta_obscuration):
    
    
    subprocess.run(["scatterplane", "--focus", str(position[0]), str(position[1]), str(position[2]),
                        "--nooutput", "--lambda", str(wavelength), "--output", filename, "--alpha", str(alpha_aperture),
                        "--beta", str(beta_obscuration), "--direction", str(direction[0]), str(direction[1]), str(direction[2]),
                        "--n", "1.0", "1.4", "--kappa", "0.005", "--samples", str(num_waves)], shell=True, capture_output=True)

# run several simulations based on the number of points in the "points" array and save the corresponding cw files in the output directory
# output_directory - location to store the output files
# points - N x 3 array storing the positions of the focal points to be simulated
# THE REST OF THE PARAMETERS MATCH simulate_point
def simulate_points(output_directory, X, Y, Z, direction, wavelength, num_waves, alpha_aperture, beta_obscuration):
    
    for i in tqdm(range(len(X))):                                                     # for each point in the focal spot
        filename = output_directory + str(i) + ".cw"
        simulate_point(filename, (X[i], Y[i], Z[i]), direction, wavelength, num_waves, alpha_aperture, beta_obscuration)

# convert cw files to sampled npy files representing the field and the desired points
def sample_points(directory, span, resolution):
    files = glob.glob(directory + "*.cw")
    for fi in tqdm(range(len(files))):
        subprocess.run(
                ["scatterview", "--input", files[fi], "--output", files[fi] + ".npy", "--size", str(span), "--nogui", "--slice", "0", "--axis", "1",
                 "--resolution", str(resolution)], shell=True, capture_output=True)

def sum_intensity(input_directory, span, resolution):
    files = glob.glob(input_directory + "*.cw")
    I = 0
    for fi in tqdm(range(len(files))):
        E = np.load(files[fi] + ".npy")
        if fi == 0:
            I = intensity(E)
        else:
            I = I + intensity(E)            
    return I



# output_directory = "C:/Users/david/Desktop/penetration_tests/"
output_directory = "D:\\myGit\\build\\scattersim\\tmp\\"
# Clean the folder
if not os.listdir(output_directory):
    print(" The root directory is clean.")
else:
    files = glob.glob(output_directory +"*")
    for file in files:
        os.remove(file)
    print("The old files in the directory are deleted.")
wavelength = 0.28
waves = 1000
direction = [1, 0, 1]
NA = 0.2
resolution = 8
span = 80
theta, phi, r = cart2sph(direction[0], direction[1], direction[2])

px, py, pz = generate_points((20, 0, 1), (60, 1, 2), phi)
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(px, py, pz)

simulate_points(output_directory, px, py, pz, direction, wavelength, waves, np.sin(NA), 0)
sample_points(output_directory, span, resolution)
I = sum_intensity(output_directory, span, resolution)

plt.imshow(I, extent=(-span/2, span/2, span/2, -span/2))
plt.set_cmap("afmhot")
plt.colorbar()
# plt.title("Intensity")
plt.show()
