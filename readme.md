# Scatter Simulation
The project aims to simulate the interactions between light and samples. The algorithm is based on Gauss' Equation, boundary conditions, and wave propagation theory.

Scatterplane solves field for homogeneous sample boundaries. Scatterthinvolume solves field for 2D heterogeneous samples. While the developing Scattervolume will solve field for 3D heterogeneous samples. Scatterview do visualization the field from the other three projects.

## Main Tools
#### Eigen (See https://eigen.tuxfamily.org/index.php?title=Main_Page)
#### Intel-MKL (See https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html)
#### CUDA (https://developer.nvidia.com/cuda-toolkit)
#### OpenGL (https://www.opengl.org/)

## Installation
[To be updated...] Vcpkg should be able to install all packages the project needs.


## Workflow
### .py files in data/ generate different samples in the format of .npy. 
### Select a proper executable from scatterplane, scatterthinvolume, and scattervolume. The calculated field will be saved as .cw file.
### Visualize the field using scatterview.

*The users can set all they need from the command line via Boost library.*

*"excutable - -help" will provide more information about the input and output.*

