import matplotlib.pyplot as plt
import numpy as np
import skimage as ski


def SampleFiber(start_pt, direction, curve, t):
    
    p =  start_pt + t * direction + (4 * t - 4 * t * t) * curve
    return p


def GenerateCenterlines(num, start, direction, step, curve, space, grid):
    
    N = grid.shape[0]
    Z = grid.shape[2]
    T = N * 4
    
    dt = (1.0 / T)
    for fi in range(num):
        # generate the start point for the fiber
        s = start + fi * space * step
        
        for ti in range(T):   
            
            
            # calculate the current time step
            t = dt * ti
            
            
            p = SampleFiber(s, direction, curve, t)
            pi_x = int(p[0] * N)
            pi_y = int(p[1] * N)
            pi_z = int(p[2] * Z)
            
            grid[pi_x, pi_y, pi_z] = 1.0
            
    return grid

def AddSphere(position, grid):
    
    p = (position[0] * grid.shape[0], position[1] * grid.shape[1])
    x = range(grid.shape[0])
    y = range(grid.shape[1])
    z = range(grid.shape[2])
    
    X, Y, Z = np.meshgrid(x, y, z)
    
    R = np.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - grid.shape[2] / 2) ** 2)
    radius = grid.shape[2] / 3
    grid[R <= radius] = 1
    
    return grid
    
    

def SaveStack(image, filemask):
    
    #image = image / np.max(image)
    for zi in range(image.shape[2]):
        filename = filemask + "_%03d" % zi + ".png"
        ski.io.imsave(filename, (image[:, :, zi] * 255).astype(np.uint8))


spacing = 0.04

N = 64
F = 5
Z = 10

UPSCALE = 16

Nu = N * UPSCALE
Zu = Z * UPSCALE

gaussian_sigma = 5

IMAGE = np.zeros((Nu, Nu, Zu))

batch_0_start = np.array([0.0, 0.15, 0.0])
batch_0_direction = np.array([1.0, 0.0, 0.0])
batch_0_step = np.array([0.0, 1.0, 0.0])
batch_0_curve = np.array([0.0, -0.1, 0.7])
IMAGE = GenerateCenterlines(F, batch_0_start, batch_0_direction, batch_0_step, batch_0_curve, spacing, IMAGE)

batch_1_start = np.array([0.0, batch_0_start[1] + batch_0_step[1] * spacing * F, 0.99]) 
batch_1_direction = batch_0_direction
batch_1_step = batch_0_step
batch_1_curve = np.array([0.0, -0.1, -0.7])
IMAGE = GenerateCenterlines(F, batch_1_start, batch_1_direction, batch_1_step, batch_1_curve, spacing, IMAGE)

RESULT = ski.filters.gaussian(IMAGE, sigma=(gaussian_sigma, gaussian_sigma, 0), mode="wrap")
RESULT = ski.filters.gaussian(RESULT, sigma=(0, 0, gaussian_sigma), mode="constant")
RESULT = RESULT / np.max(RESULT)

RESULT = AddSphere((0.6, 0.1, int(RESULT.shape[2]/2)), RESULT)
RESULT = AddSphere((0.7, 0.3, int(RESULT.shape[2]/2)), RESULT)
RESULT = AddSphere((0.6, 0.5, int(RESULT.shape[2]/2)), RESULT)
RESULT = AddSphere((0.7, 0.7, int(RESULT.shape[2]/2)), RESULT)
RESULT = AddSphere((0.6, 0.9, int(RESULT.shape[2]/2)), RESULT)

RESULT = AddSphere((0.8, 0.1, int(RESULT.shape[2]/2)), RESULT)
RESULT = AddSphere((0.9, 0.3, int(RESULT.shape[2]/2)), RESULT)
RESULT = AddSphere((0.8, 0.5, int(RESULT.shape[2]/2)), RESULT)
RESULT = AddSphere((0.9, 0.7, int(RESULT.shape[2]/2)), RESULT)
RESULT = AddSphere((0.8, 0.9, int(RESULT.shape[2]/2)), RESULT)

DOWN = ski.transform.downscale_local_mean(RESULT, UPSCALE)
    
#plt.imshow(TEST[:, :, 5])
SaveStack(DOWN, "test")