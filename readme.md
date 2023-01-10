# Coupledwave Model

## Scatterplane

### Goal
If we have a layered sample and we want to know the field at every point in the specific area. We have two steps to get this problem done. First, use some algorithms to get the field at the boundaries. Then, use the wave propagation function to calculate the electric field at all other points.

### Problem
This executable is to solve the field for each boundary.

### Solution
We use Gauss' Equation and boundary conditions to solve the linear system. The solution for the linear system is 
We use eigen library as the default solver, which is versatile, fast, and reliable. See https://eigen.tuxfamily.org/index.php?title=Main_Page 

For better performance, we use cusolver as the accelerator to solve the linear system.

### Execution

#### The users can set all they need from the command line via Boost library. 
For output message,
```
--help 		produce help message
--log		produce a log file
--output	output filename for the coupled wave structure	(a.cw)
```
For the input parameters,
```
--lambda		incident field vacuum wavelength	(1.0)
--direction		incoming field direction	(0, 0, 1)
--ex		x component of electrical field	（0， 0）
--ey		y component of electrical field	（1， 0）
--n		real refractive index (optical path length) of all L layers	(1, 1,4, 1,4, 1,0)
--kappa		absorbance of layers 2+ (layer 1 is always 0.0)	(0.05, 0.00, 0.00)
--z		position of each layer boundary	(-3.0, 0.0, 3.0)
--alpha			angle used to focus the incident field	(1)
--beta		internal obscuration angle (for simulating reflective optics)	(0)
--na 		focus angle expressed as a numerical aperture (overrides --alpha)
--samples		number of samples (can be specified in 2 dimensions)	(64) or (64, 64)
--mode		sampling mode	(polar, montecarlo)
```
#### Neccesary classes 
tira/planewave:

`scatter()` 

`solidAnglePolar()`

`solidAngleMC()`

`struct HomogeneousLayers{z, Pr, Pt} layers`

#### Main function
Flow for creating an array of incident plane waves based on the sampling angle parameters:

* InitMatices(): Get empty A and b 
* InitLayerProperties(): Get complex refractive indices for layers
* Calculate the number of samples (# of waves according to the 1-d or 2-d definition)
* Create an array of incident plan waves based on sampling angle paramters (solidAngleMC() or solidAnglePolar)
* Allocate a coupled wave structure to store simulation results
* Then for each sample(wave),
```
do{
	Initialize;
	SetBoundaryConditions();  // 6 From E0 and the last outgoing layer.
	SetGaussianConstraints(); // 2*（L-1) 
	SetBoundaryConstraints(); // For all left

	Solve the linear system by colPivHouseholderQr(). // Based on Eigen
	mat2waves(); // Generate plan waves from the solution
	
	// Log all information about the generated wave.
	
}
```
* Save the structiure cw, the structure contains an array of waves, which contains all information for the boundaries.







 

