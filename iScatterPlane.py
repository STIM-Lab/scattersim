import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import subprocess
import glob
import time
import math
from tqdm import tqdm

def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r

# Calculates the intensity of a complex X, Y, Z, 3 vector field
def intensity(E):
    Econj = np.conj(E)                                                          # calculate the complex conjugate
    I = np.sum(E * Econj, axis=-1)
    return np.real(I)

# generate a set of points representing a rectangular focal spot
# spotsize is a tuple giving the size of the focal spot
# samples is a tuple giving the number of points used to build the focal spot
# theta is the rotation angle about the z-axis [0, 2pi]
# phi is the rotation angle about the y-axis [0, pi/2]
def generate_points_rect(spotsize, samples, phi):
    x = np.linspace(-spotsize[0] / 2, spotsize[0] / 2, samples[0])
    y = np.linspace(-spotsize[1] / 2, spotsize[1] / 2, samples[1])
    z = np.linspace(-spotsize[2] / 2, spotsize[2] / 2, samples[2])

    # calculate the coordinates of the points before rotation
    X, Y, Z = np.meshgrid(x, y, z)

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    Xn = cos_phi * X + sin_phi * Z
    Yn = Y
    Zn = -sin_phi * X + cos_phi * Z

    return (np.ndarray.flatten(Xn), np.ndarray.flatten(Yn), np.ndarray.flatten(Zn))

# generate a set of points representing a rectangular focal spot
# spotsize is a tuple giving the radius of the focal spot and the thickness of the cylinder
# samples is a tuple giving the number of points used to build the focal spot
# theta is the rotation angle about the z-axis [0, 2pi]
# phi is the rotation angle about the y-axis [0, pi/2]
def generate_points_cyl(spotsize, samples, phi):
    x = np.linspace(-spotsize[0], spotsize[0], samples[0])
    y = np.linspace(-spotsize[0], spotsize[0], samples[1])
    z = np.linspace(-spotsize[1] / 2, spotsize[1] / 2, samples[2])

    # calculate the coordinates of the points before rotation
    X, Y, Z = np.meshgrid(x, y, z)
    circ = X ** 2 + Y ** 2 <= spotsize[0] ** 2
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    Xn = cos_phi * X[circ] + sin_phi * Z[circ]
    Yn = Y[circ]
    Zn = -sin_phi * X[circ] + cos_phi * Z[circ]

    return (Xn, Yn, Zn)

# generate a set of points representing a rectangular focal spot
# spotsize is a tuple giving the diameter of the focal spot
# samples is a tuple giving the number of points used to build the focal spot
# theta is the rotation angle about the z-axis [0, 2pi]
# phi is the rotation angle about the y-axis [0, pi/2]
def generate_points_uniform(spotsize, thickness, samples, phi):

    radius = spotsize/2.0

    r = np.linspace(0, 1, samples[0])
    d_theta = 2 * np.pi / samples[0]
    theta = np.linspace(0, d_theta * (samples[0] - 1), samples[1])
    z = np.linspace(-thickness/2, thickness/2, samples[2])

    R, THETA, Z = np.meshgrid(radius * np.sqrt(r), theta, z)

    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

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
def simulate_point(filesubname, position, direction, wavelengths, num_waves, alpha_aperture, beta_obscuration):

    for i in range(len(wavelengths)):
        filename = filesubname + str(wavelengths[i]) +".cw"
        subprocess.run(["D:\\myGit\\build\\scattersim\\scatterplane", "--focus", str(position[0]), str(position[1]), str(position[2]),
                        "--nooutput", "--lambda", str(wavelengths[i]), "--output", filename, "--alpha", str(alpha_aperture),
                        "--beta", str(beta_obscuration), "--direction", str(direction[0]), str(direction[1]), str(direction[2]),
                        "--n", "1.0", "1.4", "--kappa", "0.005", "--samples", str(num_waves)], shell=True, capture_output=True)

# run several simulations based on the number of points in the "points" array and save the corresponding cw files in the output directory
# output_directory - location to store the output files
# points - N x 3 array storing the positions of the focal points to be simulated
# THE REST OF THE PARAMETERS MATCH simulate_point
def simulate_points(output_directory, X, Y, Z, direction, wavelengths, num_waves, alpha_aperture, beta_obscuration):

    for i in tqdm(range(len(X))):                                                     # for each point in the focal spot
        filesubname = output_directory + str(i)
        simulate_point(filesubname, (X[i], Y[i], Z[i]), direction, wavelengths, num_waves, alpha_aperture, beta_obscuration)

# convert cw files to sampled npy files representing the field and the desired points
def sample_points(directory, simulation_range, resolution, axis=1):
    files = glob.glob(directory + "*.cw")
    for fi in tqdm(range(len(files))):
        subprocess.run(
                ["D:\\myGit\\build\\scattersim\\scatterview", "--input", files[fi], "--output", files[fi] + ".npy", "--size", str(simulation_range), "--nogui", "--slice", "0", "--axis", str(axis),
                 "--resolution", str(resolution)], shell=True, capture_output=True)


def sum_intensity(input_directory, resolution):
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
# wavelength = 0.28
wavelengths = np.linspace(0.21, 0.49, 5)
waves = 1000
points = (20, 20, 1)
total_pts = points[0] * points[1] * points[2]
direction = [1, 0, 1]
NA = 0.2
NAo = 0
resolution = 8
spotsize = (10, 10, 1)
simulation_range = spotsize[0] * 4
theta, phi, r = cart2sph(direction[0], direction[1], direction[2])
# adjust the elevation so that it corresponds to radians *from* the z axis
phi = np.pi/2.0 - phi
axis = 1

#px, py, pz = generate_points_rect(spotsize, points, phi)
radius_cyl = 5
height_cyl = 2
px, py, pz = generate_points_cyl((radius_cyl, height_cyl), points, phi)
# px, py, pz = generate_points_rect(spotsize, points, phi)

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(px, py, pz)

simulate_points(output_directory, px, py, pz, direction, wavelengths, waves, np.arcsin(NA), np.arcsin(NAo))
sample_points(output_directory, simulation_range, resolution, axis)
I = sum_intensity(output_directory, resolution)
out_File = output_directory + "figure.npy"
np.save(out_File, I)

# Visualize
# spotsize = (10, 10, 1)
# simulation_range = spotsize[0] * 4
# output_directory = "D:\\myGit\\build\\scattersim\\tmp\\"
# I = np.load(output_directory + "figure.npy")
fmt = lambda x, pos:'{:.2f}'.format(x)

fig, ax = plt.subplots(2, gridspec_kw={'height_ratios':[2, 1], 'hspace':0.3}, figsize = (6, 6))
im1 = ax[0].imshow(I, extent=(-simulation_range/2, simulation_range/2, simulation_range/2, -simulation_range/2))
im1.set_cmap("afmhot")
plt.xticks(np.arange(-simulation_range//2, simulation_range//2+1, simulation_range//4))
plt.yticks(np.arange(-simulation_range//2, simulation_range//2+1, simulation_range//4))
ticks1 = np.linspace(np.min(I), np.max(I), 5, endpoint=True)
cb = fig.colorbar(im1, ticks=ticks1, fraction=0.046, pad=0.04, format = FuncFormatter(fmt), ax = ax[0], aspect=15)

i = I[len(I)//2:len(I)//4*3, len(I)//4:len(I)//4*3]
im2 = ax[1].imshow(i, extent=(-simulation_range/4, simulation_range/4, simulation_range/4, 0))
im2.set_cmap("afmhot")
plt.xticks(np.arange(-simulation_range//4, simulation_range//4+1, simulation_range//8))
plt.yticks(np.arange(0, simulation_range//4+1, simulation_range//8))
ticks2 = np.linspace(np.min(i), np.max(i), 3, endpoint=True)
# ticks1 = np.linspace(np.min(inten), 15.49, 3, endpoint=True)
cb = fig.colorbar(im2, ticks=ticks2, fraction=0.046, pad=0.04, format = FuncFormatter(fmt), ax = ax[1], aspect=7.5)
plt.savefig(output_directory+'Simulation.png', dpi=300, bbox_inches='tight')
plt.show()

