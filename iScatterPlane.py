import numpy as np
import matplotlib.pyplot as plt
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
    rotate = True
    x = np.linspace(-spotsize[0]/2, spotsize[0]/2, samples[0])
    y = np.linspace(-spotsize[1]/2, spotsize[1]/2, samples[1])
    z = np.linspace(-spotsize[2]/2, spotsize[2]/2, samples[2])
    
    # calculate the coordinates of the points before rotation
    X, Y, Z = np.meshgrid(x, y, z)
    
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    
    Xn = cos_phi * X + sin_phi * Z
    Yn = Y
    Zn = -sin_phi * X + cos_phi * Z

    return (np.ndarray.flatten(Xn), np.ndarray.flatten(Yn), np.ndarray.flatten(Zn))

# generate a set of points representing a rectangular focal spot
# spotsize is a tuple giving the diameter of the focal spot
# samples is a tuple giving the number of points used to build the focal spot
# theta is the rotation angle about the z-axis [0, 2pi]
# phi is the rotation angle about the y-axis [0, pi/2]
def generate_points_circ(spotsize, thickness, samples, phi):
    
    radius = spotsize/2.0
    
    r = radius * np.sqrt(np.random.random(samples))
    theta = np.random.random(samples) * 2 * np.pi
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.random(samples) * thickness - thickness/2
    
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    
    xn = cos_phi * x + sin_phi * z
    yn = y
    zn = -sin_phi * x + cos_phi * z
    

    return (xn, yn, zn)

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
       
        

def sum_intensity(directory, simulation_range, resolution, axis=1):
    subprocess.run(
            ["scatterview", "--input", directory + "*.cw", "--output", directory + "result.npy", "--size", str(simulation_range), "--nogui", "--slice", "0", "--axis", str(axis),
             "--resolution", str(resolution), "--intensity"], shell=True, capture_output=True)
    Intensity = np.load(directory + "result.npy")
    return Intensity


output_directory = "C:/Users/david/Documents/intensity_test/"
#output_directory = "C:/Users/david/Desktop/penetration_tests/"
#output_directory = "D:\\myGit\\build\\scattersim\\tmp\\"
# Clean the folder
if not os.listdir(output_directory):
    print(" The root directory is clean.")
else:
    files = glob.glob(output_directory +"*")
    for file in files:
        os.remove(file)
    print("The old files in the directory are deleted.")
wavelength = 0.28
waves = 4000
points = (20, 20, 1)
total_pts = points[0] * points[1] * points[2]
direction = [1, 0, 1]
NA = 0.2
NAo = 0
resolution = 8
spotsize = (10, 10, 1)
theta, phi, r = cart2sph(direction[0], direction[1], direction[2])
# adjust the elevation so that it corresponds to radians *from* the z axis
phi = np.pi/2.0 - phi
axis = 2

px, py, pz = generate_points_rect(spotsize, points, phi)
#px, py, pz = generate_points_circ(spotsize[0], spotsize[2], total_pts, phi)
#px, py, pz = generate_points_uniform(spotsize[0], spotsize[2], points, phi)

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(px, py, pz)

simulate_points(output_directory, px, py, pz, direction, wavelength, waves, np.arcsin(NA), np.arcsin(NAo))
I = sum_intensity(output_directory, spotsize[0] * 2, resolution, axis)

plt.imshow(I, extent=(-spotsize[0], spotsize[0], spotsize[0], -spotsize[0]))
plt.set_cmap("afmhot")
plt.colorbar()
# plt.title("Intensity")
plt.show()
