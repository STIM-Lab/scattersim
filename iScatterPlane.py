import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import glob
import time
import math
from tqdm import tqdm

# Creating a Function.
def normal_dist(x):
    prob_density = (np.pi * np.std(x)) * np.exp(-0.5 * ((x - np.mean(x)) / np.std(x)) ** 2)
    return prob_density

# Return the intensity of the field everywhere.
def intensity(E):
    Econj = np.conj(E)
    I = np.sum(E * Econj, axis=-1)
    return np.real(I)

def generate_points(diameter_xy, diameter_z, n_x, n_y, n_z, theta):
    points = []
    rotate = True
    radius = diameter_xy / 2
    X = np.linspace(0, diameter_xy, n_x).reshape([1, 1, n_x])
    Y = np.linspace(0, diameter_xy, n_y).reshape([1, n_y, 1])
    Z = np.linspace(0, diameter_z, n_z).reshape([n_z, 1, 1])
    # circle = (X - radius) ** 2 + (Y - radius) ** 2 <= radius ** 2
    for j in range(len(Y[0])):
        for i in range(len(X[0][0])):
            for k in range(len(Z)):
                # if circle[0][j][i] == True:
                x_pos = (i - len(X[0][0]) // 2) * diameter_xy / len(X[0][0])
                y_pos = (j - len(Y[0]) // 2) * diameter_xy / len(Y[0])
                z_pos = (k - len(Z) // 2) * diameter_z / len(Z)

                if rotate == True:
                    # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
                    x_new_pos = math.cos(theta) * x_pos + math.sin(theta) * z_pos
                    z_new_pos = -math.sin(theta) * x_pos + math.cos(theta) * z_pos
                    points.append([x_new_pos, y_pos, z_new_pos])
    print("How many points do we have? --" + str(len(points) * n_lambda))
    return points
def simulate_point(points, direction, wavelengths, output_directory, output_fig, n_CoherentWaves, size, resolution, alpha, obscuration):
    # Simulate the fields and sum them up.
    num = 0
    # Run "scatterplane" n_lambda * len(points) * n_samples times. Generate len(points) ".npy" files
    for w in range(len(wavelengths)):
        Intensity = 0
        for i in tqdm(range(len(points))):
            in_str = output_directory + str(wavelengths[w]) + str(points[i][0]) + str(points[i][1]) + str(
                points[i][2]) + ".cw"
            out_str = output_directory + str(wavelengths[w]) + str(points[i][0]) + str(points[i][1]) + str(
                points[i][2]) + ".npy"

            proc = subprocess.run(["scatterplane", "--focus", str(points[i][0]), str(points[i][1]), str(points[i][2]),
                                   "--nooutput", "--lambda", str(wavelengths[w]), "--output", in_str, "--alpha",
                                   str(alpha),
                                   "--beta", str(obscuration), "--direction", str(direction[0]), str(direction[1]),
                                   str(direction[2]),
                                   "--n", "1.0", "1.4", "--kappa", "0.005", "--samples", str(n_CoherentWaves), "--wavemask",
                                   str(1), str(1), str(1)], shell=True, capture_output=True)

            proc = subprocess.run(
                ["scatterview", "--input", in_str, "--output", out_str, "--size", str(size), "--nogui", "--slice", "0",
                 "--axis", "1",
                 "--resolution", str(resolution)], shell=True, capture_output=True)

            num += 1
            inFile = os.path.join(output_directory, out_str)
            E = np.load(inFile)
            if w == 0 and i == 0:
                Intensity = abs(intensity(E))
            else:
                Intensity += abs(intensity(E))
    print(num)
    print("What is the number of the total waves? -- " + str(num))
    # os.remove(inFile)
    # os.remove(os.path.join(output_directory, in_str))
    np.save(output_fig, Intensity)
    return Intensity


start = time.time()
# Clean the current folder
output_directory = "D:\\myGit\\build\\scattersim\\tmp\\"
if not os.listdir(output_directory):
    print(" The root directory is clean.")
else:
    files = glob.glob(output_directory +"*")
    for file in files:
        os.remove(file)
    print("The old files in the directory are deleted.")

# Define background information for simulation
direction = [0, 0, 1]  # Central direction
# direction = [1, 0, 1/np.tan(math.radians(70))]
wavelength_0 = 0.28     # The central wavelength
n_lambda = 5           # The number of wavelengths
wavelengths = np.linspace(wavelength_0-0.05, wavelength_0+0.05, n_lambda)

# Define the optical system
fname = "global"
output_fig = output_directory + fname + ".npy"
n_CoherentWaves = 200        # Number of coupled waves for each point
size = 80               # The size of the canvas (um)
resolution = 9
alpha = 0.5              # Set the radian
obscuration = alpha * 0.18    # Set percentage

# Generate point sources
diameter_xy = 20
diameter_z = wavelength_0 * 2
n_x = 10                # Sampling points along x
n_y = 1                 # Sampling points along y
n_z = 1                  # Sampling points along z
theta = np.arctan(direction[0] / direction[2])
points = generate_points(diameter_xy, diameter_z, n_x, n_y, n_z, theta)

Intensity = simulate_point(points, direction, wavelengths, output_directory, output_fig, n_CoherentWaves, size, resolution, alpha, obscuration)

# Visualization
plt.imshow(Intensity, extent=(-size/2, size/2, size/2, -size/2))
plt.set_cmap("afmhot")
plt.colorbar(fraction=0.023, pad=0.3)
# plt.title("Intensity")
plt.show()

# inten is part of Intensity. It is for extracting the uniform area and calculating the DOE
inten = Intensity[len(Intensity)//2:len(Intensity)//4*3, len(Intensity)//4:len(Intensity)//4*3]
plt.imshow(inten, extent=(-size/4, size/4, size//4, 0))
plt.set_cmap("afmhot")
plt.colorbar(fraction=0.023, pad=0.3)
# plt.title("Intensity")
plt.show()
end = time.time()
print("Total time is: " + str(end - start) + " s.")


# Calculate the penetration depth
target = 0
threshold = np.mean(inten[target]) * 0.37
for i in range(target+1, len(inten)):
    if np.mean(inten[i]) < threshold:
        depth = (i - target) / len(inten) * size/4
        print("The penetration depth is: " + str(depth))
        break