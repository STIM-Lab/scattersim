#include <iostream>
#include <tira/optics/planewave.h>
#include "CoupledWaveStructure.h"
#include "FourierWave.h"
#include <complex>
#include <string>
#include <math.h>
#include <fstream>
#include <boost/program_options.hpp>
#include <random>
#include <iomanip>
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include <extern/libnpy/npy.hpp>
#include <chrono> 
#include <ctime>


#include "tira/optics/planewave.h"
#include "tira/field.h"
#include "third_Lapack.h"

// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif

std::vector<double> in_dir;
double in_lambda;

std::vector<double> in_n;
std::vector<double> in_kappa;
std::vector<double> in_ex;
std::vector<double> in_ey;
std::vector<double> in_ez;
double in_z;
std::vector<double> in_size;
std::vector<size_t> num_pixels;
std::vector<double> in_normal;
//double in_na;
std::string in_outfile;
//double in_alpha;
//double in_beta;
std::string in_sample;
//std::string in_mode;
std::vector<int> in_coeff;
double in_n_sample;
double in_kappa_sample;
std::vector<double> in_center;
std::vector<double> in_rec_bar;
std::vector<double> in_circle;

unsigned int L;
int M[2];	// Set the number of the Fourier Coefficients
Eigen::MatrixXcd A;
Eigen::VectorXcd b;
Eigen::VectorXcd ni;
double* z;
std::complex<double> k;
std::vector<std::complex<double>> E0;
Eigen::VectorXcd EF;
int MF;
Eigen::RowVectorXcd Sx;				// Fourier coefficients of x component of direction
Eigen::RowVectorXcd Sy;				// Fourier coefficients of y component of direction
std::vector<Eigen::RowVectorXcd> Sz(2);		// 2D vector for the Fourier coefficients of z component for the upper and lower regions
Eigen::VectorXcd Ex, Ey, Ez;
int ei = 0;				// The current row for the matrix
int l;			// The current layer l.
std::ofstream logfile;
std::ofstream proffile;
std::vector<Eigen::MatrixXcd> D;		// The property matrix
Eigen::VectorXcd evl;
Eigen::MatrixXcd evt;
std::vector<Eigen::VectorXcd> eigenvalues;			// eigen values for current layer
std::vector<Eigen::MatrixXcd> eigenvectors;			// eigen vectors for current layer
std::vector<Eigen::VectorXcd> Beta;			// eigen vectors for current layer
Eigen::MatrixXcd Gc;					// Upward
Eigen::MatrixXcd Gd;					// Downward
std::vector<Eigen::VectorXcd > Eigenvalues;			// Dimension: (layers, coeffs)
std::vector<Eigen::MatrixXcd> Eigenvectors;			// Dimension: (layers, coeffs)
std::vector<Eigen::MatrixXcd> GD;					// Dimension: (layers, coeffs)
std::vector<Eigen::MatrixXcd> GC;					// Dimension: (layers, coeffs)
Eigen::MatrixXcd f1;
Eigen::MatrixXcd f2;
Eigen::MatrixXcd f3;
Eigen::MatrixXcd tmp;					// Temposarily store some Eigen::MatrixXcd
Eigen::MatrixXcd tmp_2;					// Temposarily store additional Eigen::MatrixXcd
Eigen::MatrixXcd Gc_static;					// Temposarily store some Eigen::MatrixXcd
std::chrono::duration<double> elapsed_seconds;
bool saveSampleTexture = true;
int counts = 0;

/// Convert a complex vector to a string for display
template <typename T>
std::string vec2str(glm::vec<3, std::complex<T> > v, int spacing = 20) {
	std::stringstream ss;
	if (v[0].imag() == 0.0 && v[1].imag() == 0.0 && v[2].imag() == 0.0) {				// if the vector is real
		ss << std::setw(spacing) << std::left << v[0].real() << std::setw(spacing) << std::left << v[1].real() << std::setw(spacing) << std::left << v[2].real();
	}
	else {
		ss << std::setw(spacing) << std::left << v[0] << std::setw(spacing) << std::left << v[1] << std::setw(spacing) << std::left << v[2];
	}
	return ss.str();
}

/// Convert a real vector to a string for display
std::string vec2str(glm::vec<3, double> v, int spacing = 20) {
	std::stringstream ss;
	ss << std::setw(spacing) << std::left << v[0] << std::setw(spacing) << std::left << v[1] << std::setw(spacing) << std::left << v[2];
	return ss.str();
}

/// Return a value in the A matrix
std::complex<double>& Mat(int row, int col) {
	return A(row, col);
}

// Enumerators used to access elements of the A matrix
enum Coord { X, Y, Z };
enum Dir { Transmitted, Reflected };

// Methods used to access elements of the matrix A based on the layer number, direction, and field coordinate
std::complex<double>& Mat(int row, int layer, Dir d, Coord c, int m, int M) {
	return A(row, (layer * 6 + d * 3 + c - 3) * M + m);
}
size_t idx(int layer, Dir d, Coord c, int m, int M) {
	return (layer * 6 + d * 3 + c - 3) * M + m;
}

/// Output the coupled wave matrix as a string
std::string mat2str(int width = 10, int precision = 2) {
	std::stringstream ss;
	ss << A;
	return ss.str();
}

// Output the b matrix (containing the unknowns) to a string
std::string b2str(int precision = 2) {
	std::stringstream ss;
	ss << b;
	return ss.str();
}

// Initialize the A matrix and b vector of unknowns
void InitMatrices() {
	A = Eigen::MatrixXcd::Zero(6 * MF * (L - 1), 6 * MF * (L - 1));									// allocate space for the matrix
	b = Eigen::VectorXcd::Zero(6 * MF * (L - 1));												// zero out the matrix
}

// set all of the layer refractive indices and boundary positions based on user input
void InitLayerProperties() {
	ni.resize(L);
	ni[0] = std::complex<double>(in_n[0], 0);
	for (size_t l = 1; l < L; l++)
		ni[l] = std::complex<double>(in_n[l], in_kappa[l - 1]);			// store the complex refractive index for each layer
	z = new double[L];														// allocate space to store z coordinates for each interface
	z[0] = in_z - in_size[2] / 2.0;
	z[1] = in_z + in_size[2] / 2.0;
}

// The struct is to integrate eigenvalues and their indices
struct EiV {
	size_t idx;
	std::complex<double> value;
};

// Sort by eigenvalues' imaginery parts. The real parts are the tie-breaker.
bool sorter(EiV const& lhs, EiV const& rhs) {
	if (lhs.value.imag() != rhs.value.imag())
		return lhs.value.imag() < rhs.value.imag();
	else if (lhs.value.real() != rhs.value.real())
		return lhs.value.real() < rhs.value.real();
	else
		return lhs.value.imag() < rhs.value.imag();
}
/// <summary>
/// Temporarily depreacated.
/// <summary>
/// <param name="eigenvalues_unordered"></param>
/// <param name="eigenvectors_unordered"></param>
void Eigen_Sort(Eigen::VectorXcd eigenvalues_unordered, Eigen::MatrixXcd eigenvectors_unordered) {
	unsigned int len = eigenvalues_unordered.size();
	// Sort the unordered eigenvalues and track the indices
	std::vector<EiV> eiV(len);
	for (size_t i = 0; i < len; i++) {
		eiV[i].idx = i;
		eiV[i].value = eigenvalues_unordered(i);
	}
	std::sort(eiV.begin(), eiV.end(), &sorter);

	evl.resize(len);
	evt.resize(len, len);
	//for (size_t i = 0; i < len; i++) {
	//	std::cout << "eiV: " << eiV[i].idx << std::endl;
	//}
	if (logfile) {
		logfile << "eigenvalues_unordered: " << std::endl;
		logfile << eigenvalues_unordered << std::endl << std::endl;
		logfile << "eigenvectors_unordered: " << std::endl;
		logfile << eigenvectors_unordered << std::endl << std::endl;
	}
	for (size_t i = 0; i < len / 2; i++) {
		evl[2 * i] = eigenvalues_unordered[eiV[len - 1 - i].idx];
		evt.col(2 * i) = eigenvectors_unordered.col(eiV[len - 1 - i].idx);
		evl[2 * i + 1] = eigenvalues_unordered[eiV[i].idx];
		evt.col(2 * i + 1) = eigenvectors_unordered.col(eiV[i].idx);
		//std::cout << "eigenvalues_unordered: " << eigenvalues_unordered << std::endl;
		//std::cout << "eigenvalues: " << evl << std::endl;
	}
	if (logfile) {
		logfile << "evl: " << std::endl;
		logfile << evl << std::endl << std::endl;
		logfile << "evt: " << std::endl;
		logfile << evt << std::endl << std::endl;
	}

	eigenvalues.push_back(evl);				// For computing the inner structure of the sample
	eigenvectors.push_back(evt);
}

// Do eigen decomposition for Phi. 
// Sort the eigenvectors and eigenvalues by pairs. 
// Build matrices Gd and Gc.
void EigenDecompositionD() {
	std::vector<Eigen::VectorXcd> eigenvalues_unordered;
	std::vector<Eigen::MatrixXcd> eigenvectors_unordered;
	bool EIGEN = false;
	bool MKL_lapack = true;
	GD.reserve(D.size());
	GC.reserve(D.size());
	for (size_t i = 0; i < D.size(); i++) {
		if (EIGEN) {
			Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(D[i]);
			eigenvalues_unordered.push_back(es.eigenvalues());
			eigenvectors_unordered.push_back(es.eigenvectors());
			Eigen_Sort(eigenvalues_unordered[i], eigenvectors_unordered[i]);
		}
		if (MKL_lapack) {
			std::complex<double>* A = new std::complex<double>[4 * MF * 4 * MF];
			Eigen::MatrixXcd::Map(A, D[i].rows(), D[i].cols()) = D[i];
			std::complex<double>* evl = new std::complex<double>[4 * MF];
			std::complex<double>* evt = new std::complex<double>[4 * MF * 4 * MF];
			std::chrono::time_point<std::chrono::system_clock> s = std::chrono::system_clock::now();
			MKL_eigensolve(A, evl, evt, 4 * MF);
			//const std::vector<long unsigned> shape{ (unsigned long)4 * MF, (unsigned long)4 * MF };
			//const bool fortran_order{ false };
			//npy::SaveArrayAsNumpy("A.npy", fortran_order, shape.size(), shape.data(), &D[i](0, 0));

			std::chrono::time_point<std::chrono::system_clock> e = std::chrono::system_clock::now();
			elapsed_seconds = e - s;
			proffile << "						Time for MKL_eigensolve():" << elapsed_seconds.count() << "s" << std::endl;
			eigenvalues_unordered.push_back(Eigen::Map<Eigen::VectorXcd>(evl, 4 * MF));
			eigenvectors_unordered.push_back(Eigen::Map < Eigen::MatrixXcd, Eigen::ColMajor >(evt, 4 * MF, 4 * MF));
			//std::cout << "eigenvalues_unordered: " << eigenvalues_unordered[i] << std::endl;
			Eigen_Sort(eigenvalues_unordered[i], eigenvectors_unordered[i]);
			//npy::SaveArrayAsNumpy("evt_unordered.npy", fortran_order, shape.size(), shape.data(), &eigenvectors_unordered[i](0, 0));
			//npy::SaveArrayAsNumpy("evt.npy", fortran_order, shape.size(), shape.data(), &eigenvectors[i](0, 0));
			//for (int ii = 0; ii < 400; ii++)
			//	std::cout << "eigenvalues: " << (eigenvalues[i](ii)) << std::endl;
		}

	}

	Gd.resize(4 * MF, 4 * MF);
	Gc.resize(4 * MF, 4 * MF);
	std::complex<double> Di;
	std::complex<double> Ci;
	Beta.resize(num_pixels[0]);
	std::complex<double> Di_exp;
	std::complex<double> Ci_exp;
	for (size_t i = 0; i < D.size(); i++) {
		for (size_t j = 0; j < eigenvalues[i].size(); j++) {
			if (D.size() == 1) {
				Ci = std::exp(std::complex<double>(0, 1) * k * eigenvalues[i](j) * (std::complex<double>)(in_size[2]));
				Di = std::exp(std::complex<double>(0, 1) * k * eigenvalues[i](j) * (std::complex<double>)(-in_size[2]));
			}
			else {
				/// Use the commented version when you don't need the heterogeneous info.
				/// Reason: Little phase mismatch due to the dif between simulation and physics.
				// In multi-layer case, let's suppose in_size[1] == extent is true.
				//std::cout << "eigenvalues[i](j): " << eigenvalues[i](j) << std::endl;
				Di_exp = std::complex<double>(0, 1) * k * eigenvalues[i](j) * ((std::complex<double>) (-in_size[2]) / (std::complex<double>)(num_pixels[0]));
				Ci_exp = std::complex<double>(0, 1) * k * eigenvalues[i](j) * ((std::complex<double>) (in_size[2]) / (std::complex<double>)(num_pixels[0]));
				Ci = std::exp(Ci_exp);
				Di = std::exp(Di_exp);
			}
			if (j % 2 != 0) {
				Gd.col(j) = eigenvectors[i].col(j) * Di;
				Gc.col(j) = eigenvectors[i].col(j);
				if (Di_exp.real() > 0)
					std::cout << "Debug the eigen sorter again! Di" << std::endl;
			}

			else {
				Gd.col(j) = eigenvectors[i].col(j);
				Gc.col(j) = eigenvectors[i].col(j) * Ci;
				if (Ci_exp.real() > 0)
					std::cout << "Debug the eigen sorter again! Ci" << std::endl;
			}
		}
		//const std::vector<long unsigned> shape{ (unsigned long) 4 * MF, (unsigned long)4 * MF };
		//const bool fortran_order{ false };
		//npy::SaveArrayAsNumpy("Gd.npy", fortran_order, shape.size(), shape.data(), &Gd(0, 0));      
		//npy::SaveArrayAsNumpy("Gc.npy", fortran_order, shape.size(), shape.data(), &Gc(0, 0));
		//npy::SaveArrayAsNumpy("eigenvectors.npy", fortran_order, shape.size(), shape.data(), &eigenvectors[0](0, 0));
		GD.push_back(Gd);
		GC.push_back(Gc);
		if (i == 0)
			Gc_static = Gc;
		else {
			tmp = MKL_inverse(Gd);
			tmp_2 = MKL_multiply(Gc, tmp, 1);

			//const std::vector<long unsigned> shape{ (unsigned long)4 * MF, (unsigned long)4 * MF };
			//const bool fortran_order{ false };
			//Eigen::MatrixXcd A_block2 = MKL_multiply(tmp, f3, 1);
			//npy::SaveArrayAsNumpy("Gc.npy", fortran_order, shape.size(), shape.data(), &Gc(0, 0));
			//npy::SaveArrayAsNumpy("Gd_inv.npy", fortran_order, shape.size(), shape.data(), &tmp(0, 0));

			Gc = MKL_multiply(tmp_2, Gc_static, 1);
			Gc_static = Gc;
		}
		if (logfile) {
			logfile << "----------For the layer---------- " << std::endl;
			logfile << "Property matrix D: " << std::endl;
			logfile << D[i] << std::endl;
			logfile << "GD: " << std::endl;
			logfile << GD[i] << std::endl;
			logfile << "GD inversese: " << std::endl;
			logfile << Gd.inverse() << std::endl;
			logfile << "GC: " << std::endl;
			logfile << GC[i] << std::endl;
			logfile << "GC inversese: " << std::endl;
			logfile << Gd.inverse() << std::endl;
		}
	}
}

void MatTransfer() {
	f1.resize(4 * MF, 3 * MF);
	f2.resize(4 * MF, 3 * MF);
	f3.resize(4 * MF, 3 * MF);
	f1.setZero();
	f2.setZero();
	f3.setZero();

	// Focus on z=0
	Eigen::RowVectorXcd phase = (std::complex<double>(0, 1) * k * (std::complex<double>)(z[0]) * Eigen::Map<Eigen::RowVectorXcd>(Sz[0].data(), Sz[0].size())).array().exp();
	Eigen::MatrixXcd Phase = phase.replicate(MF, 1);		// Phase is the duplicated (by row) matrix from phase.

	Eigen::MatrixXcd SZ0 = Sz[0].replicate(MF, 1);		// neg_SZ0 is the duplicated (by row) matrix from neg_Sz0.
	Eigen::MatrixXcd SZ1 = Sz[1].replicate(MF, 1);		// neg_SZ0 is the duplicated (by row) matrix from neg_Sz0.
	Eigen::MatrixXcd SX = Sx.replicate(MF, 1);		// neg_SX is the duplicated (by row) matrix from phase.
	Eigen::MatrixXcd SY = Sy.replicate(MF, 1);		// neg_SX is the duplicated (by row) matrix from phase.
	Eigen::MatrixXcd identity = Eigen::MatrixXcd::Identity(MF, MF);

	// first constraint (Equation 8)
	f1.block(0, 0, MF, MF) = identity.array() * Phase.array();
	f1.block(MF, MF, MF, MF) = identity.array() * Phase.array();
	f1.block(2 * MF, MF, MF, MF) = (std::complex<double>(-1, 0)) * identity.array() * Phase.array() * SZ0.array();
	f1.block(2 * MF, 2 * MF, MF, MF) = identity.array() * Phase.array() * SY.array();
	f1.block(3 * MF, 0, MF, MF) = identity.array() * Phase.array() * SZ0.array();
	f1.block(3 * MF, 2 * MF, MF, MF) = (std::complex<double>(-1, 0)) * identity.array() * Phase.array() * SX.array();

	// second constraint (Equation 9)
	f2.block(0, 0, MF, MF) = identity.array();
	f2.block(MF, MF, MF, MF) = identity.array();
	f2.block(2 * MF, MF, MF, MF) = SZ0.array() * identity.array();
	f2.block(2 * MF, 2 * MF, MF, MF) = SY.array() * identity.array();
	f2.block(3 * MF, 0, MF, MF) = std::complex<double>(-1, 0) * SZ0.array() * identity.array();
	f2.block(3 * MF, 2 * MF, MF, MF) = std::complex<double>(-1, 0) * SX.array() * identity.array();

	// third constraint (Equation 10)	
	f3.block(0, 0, MF, MF) = -identity.array();
	f3.block(MF, MF, MF, MF) = -identity.array();
	f3.block(2 * MF, MF, MF, MF) = SZ1.array() * identity.array();
	f3.block(2 * MF, 2 * MF, MF, MF) = -SY.array() * identity.array();
	f3.block(3 * MF, 0, MF, MF) = -SZ1.array() * identity.array();
	f3.block(3 * MF, 2 * MF, MF, MF) = SX.array() * identity.array();

	if (logfile) {
		logfile << "f1: " << std::endl;
		logfile << f1 << std::endl;
		logfile << "f2: " << std::endl;
		logfile << f2 << std::endl;
		logfile << "f3: " << std::endl;
		logfile << f3 << std::endl;
	}
}

//// Set the equations that force the divergence of the electric field to be zero (Gauss' equation)
void SetGaussianConstraints() {

	// set reflected constraints
	for (size_t m = 0; m < MF; m++) {
		Mat(ei, 0, Reflected, X, m, MF) = Sx(m);
		Mat(ei, 0, Reflected, Y, m, MF) = Sy(m);
		Mat(ei, 0, Reflected, Z, m, MF) = -Sz[0](m);
		ei += 1;
	}
	// set transmitted constraints
	for (size_t m = 0; m < MF; m++) {
		Mat(ei, 1, Transmitted, X, m, MF) = Sx(m);
		Mat(ei, 1, Transmitted, Y, m, MF) = Sy(m);
		Mat(ei, 1, Transmitted, Z, m, MF) = Sz[1](m);
		ei += 1;
	}
}

// Force the field within each layer to be equal at the layer boundary
void SetBoundaryConditions() {
	std::complex<double> i(0.0, 1.0);
	proffile << "		Boundary conditions setting starts..." << std::endl;
	std::chrono::time_point<std::chrono::system_clock> eigen1 = std::chrono::system_clock::now();
	EigenDecompositionD();		// Compute GD and GC
	std::chrono::time_point<std::chrono::system_clock> eigen2 = std::chrono::system_clock::now();
	elapsed_seconds = eigen2 - eigen1;
	proffile << "			Time for EigenDecompositionD(): " << elapsed_seconds.count() << "s" << std::endl;

	MatTransfer();				// Achieve the connection between the variable vector and the field vector
	std::chrono::time_point<std::chrono::system_clock> matTransfer = std::chrono::system_clock::now();
	elapsed_seconds = matTransfer - eigen2;
	proffile << "			Time for MatTransfer(): " << elapsed_seconds.count() << "s" << std::endl;

	//const std::vector<long unsigned> shape{ (unsigned long)4 * MF, (unsigned long)4 * MF };
	//const bool fortran_order{ false };
	//Eigen::MatrixXcd A_block2 = MKL_multiply(tmp, f3, 1);
	//npy::SaveArrayAsNumpy("Gc_static.npy", fortran_order, shape.size(), shape.data(), &Gc_static(0, 0));

	Eigen::MatrixXcd Gc_inv = MKL_inverse(Gc_static);
	std::chrono::time_point<std::chrono::system_clock> inv = std::chrono::system_clock::now();
	elapsed_seconds = inv - matTransfer;
	proffile << "			Time for MKL_inverse(): " << elapsed_seconds.count() << "s" << std::endl;

	A.block(2 * MF, 0, 4 * MF, 3 * MF) = f2;
	tmp = MKL_multiply(GD[0], Gc_inv, 1);

	//const std::vector<long unsigned> shape{ (unsigned long)4 * MF, (unsigned long)4 * MF };
	//const bool fortran_order{ false };
	//Eigen::MatrixXcd A_block2 = MKL_multiply(tmp, f3, 1);
	//npy::SaveArrayAsNumpy("GD[0].npy", fortran_order, shape.size(), shape.data(), &GD[0](0, 0));
	//npy::SaveArrayAsNumpy("Gc_inv.npy", fortran_order, shape.size(), shape.data(), &Gc_inv(0, 0));

	A.block(2 * MF, 3 * MF, 4 * MF, 3 * MF) = MKL_multiply(tmp, f3, 1);
	std::chrono::time_point<std::chrono::system_clock> mul = std::chrono::system_clock::now();
	elapsed_seconds = mul - inv;
	proffile << "			Time for multiplication once: " << elapsed_seconds.count() / 2 << "s" << std::endl;

	b.segment(2 * MF, 4 * MF) = std::complex<double>(-1, 0) * f1 * Eigen::Map<Eigen::VectorXcd>(EF.data(), 3 * MF);

	if (logfile) {
		logfile << "LHS matrix in the linear system:" << std::endl;
		logfile << A << std::endl << std::endl;
		logfile << "RHS vector in the linear system:" << std::endl;
		logfile << b << std::endl << std::endl;
	}
}

// Converts a b vector to a list of corresponding plane waves
std::vector<tira::planewave<double>> mat2waves(tira::planewave<double> i, Eigen::VectorXcd x, size_t p) {
	std::vector<tira::planewave<double>> P;

	P.push_back(i);											// push the incident plane wave into the P array

	tira::planewave<double> r(Sx(p) * k,
		Sy(p) * k,
		-Sz[0](p) * k *in_n[0],
		x[idx(0, Reflected, X, p, MF)],
		x[idx(0, Reflected, Y, p, MF)],
		x[idx(0, Reflected, Z, p, MF)]
	);
	tira::planewave<double> t(Sx(p) * k,
		Sy(p) * k,
		Sz[1](p) * k * in_n[1],
		x[idx(1, Transmitted, X, p, MF)],
		x[idx(1, Transmitted, Y, p, MF)],
		x[idx(1, Transmitted, Z, p, MF)]
	);
	//std::cout << "r: " << std::endl << r.str() << std::endl;
	//std::cout << "t: " << std::endl << t.str()<< std:: endl;
	P.push_back(r);
	P.push_back(t);
	return P;
}

/// Removes waves in the input set that have a k-vector pointed along the negative z axis
std::vector< tira::planewave<double> > RemoveInvalidWaves(std::vector<tira::planewave<double>> W) {
	std::vector<tira::planewave<double>> new_W;
	for (size_t i = 0; i < W.size(); i++) {
		if (W[i].getKreal()[2] > 0)
			new_W.push_back(W[i]);
	}

	return new_W;
}

int main(int argc, char** argv) {
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

	// Set up all of the input options provided to the user
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("sample", boost::program_options::value<std::string>(&in_sample), "input sample as an .npy file")
		("lambda", boost::program_options::value<double>(&in_lambda)->default_value(1.0), "incident field vacuum wavelength")
		("direction", boost::program_options::value<std::vector<double> >(&in_dir)->multitoken()->default_value(std::vector<double>{0, 0, 1}, "0, 0, 1"), "incoming field direction")
		("ex", boost::program_options::value<std::vector<double> >(&in_ex)->multitoken()->default_value(std::vector<double>{0, 0}, "0, 0"), "x component of the electrical field")
		("ey", boost::program_options::value<std::vector<double> >(&in_ey)->multitoken()->default_value(std::vector<double>{1, 0}, "1, 0"), "y component of the electrical field")
		("ez", boost::program_options::value<std::vector<double> >(&in_ez)->multitoken()->default_value(std::vector<double>{0, 0}, "0 0"), "z component of the electrical field")
		("n", boost::program_options::value<std::vector<double>>(&in_n)->multitoken()->default_value(std::vector<double>{1.0, 1.0}, "1, 1"), "real refractive index (optical path length) of the upper and lower layers")
		("kappa", boost::program_options::value<std::vector<double> >(&in_kappa)->multitoken()->default_value(std::vector<double>{0}, "0.00"), "absorbance of the lower layer (upper layer is always 0.0)")
		// The center of the sample along x/y is always 0/0.
		("size", boost::program_options::value<std::vector<double>>(&in_size)->multitoken()->default_value(std::vector<double>{10, 10, 4}, "20, 20, 10"), "The real size of the single-layer sample")
		("z", boost::program_options::value<double >(&in_z)->multitoken()->default_value(0, "0.0"), "the center of the sample along z-axis")
		("output", boost::program_options::value<std::string>(&in_outfile)->default_value("c.cw"), "output filename for the coupled wave structure")
		("coef", boost::program_options::value<std::vector<int> >(&in_coeff)->multitoken(), "number of Fourier coefficients (can be specified in 2 dimensions)")
		("log", "produce a log file")
		("prof", "produce a profiling file")
		// input just for scattervolume 
		;
	// I have to do some strange stuff in here to allow negative values in the command line. I just wouldn't change any of it if possible.
	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).style(
		boost::program_options::command_line_style::unix_style ^ boost::program_options::command_line_style::allow_short
	).run(), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {									// output all of the command line options
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("prof")) {									// if a log is requested, begin output
		std::stringstream ss;
		ss << std::time(0) << "_scattervolume.prof";
		proffile.open(ss.str());
	}

	proffile << "Initialization starts..." << std::endl;

	if (vm.count("log")) {									// if a log is requested, begin output
		std::stringstream ss;
		ss << std::time(0) << "_scattervolume.log";
		logfile.open(ss.str());
	}

	// Calculate the number of layers based on input parameters (take the maximum of all layer-specific command-line options)
	L = in_n.size();
	
	Eigen::Vector3d dir(in_dir[0], in_dir[1], in_dir[2]);
	dir.normalize();																							// set the normalized direction of the incoming source field

	glm::tvec3<std::complex<double>> e = glm::tvec3<std::complex<double>>(std::complex<double>(in_ex[0], in_ex[1]),
		std::complex<double>(in_ey[0], in_ey[1]),
		std::complex<double>(in_ez[0], in_ez[1]));				// set the input electrical field
	glm::tvec3<double> dirvec(dir(0), dir(1), dir(2));
	orthogonalize(e, dirvec);
	dir = dir * in_n[0];

	// wavenumber
	k = (std::complex<double>)(2 * PI / in_lambda);

	// store all of the layer positions and refractive indices
	InitLayerProperties();
	// Define sample volume, reformat, and reorgnize.
	volume < std::complex< double> > Volume(in_sample, ni, z, in_center, in_size, k.real(), std::complex<double>(in_n_sample, in_kappa_sample));
	num_pixels = Volume.reformat();

	// Get the number of the Fourier Coefficients
	if (in_coeff.size() == 0) {
		M[0] = num_pixels[2];
		M[1] = num_pixels[1];
	}
	else if (in_coeff.size() == 1) {
		M[0] = std::sqrt(in_coeff[0]);
		M[1] = std::sqrt(in_coeff[0]);
	}
	else if (in_coeff.size() == 2) {
		M[0] = in_coeff[0];
		M[1] = in_coeff[1];
	}

	// Give warning if the decomposed wave goes opposite.
	if (pow((double(M[0] / 2) / in_size[0] * in_lambda + dir[0]), 2) + (pow((double(M[1] / 2) / in_size[1] * in_lambda + dir[1]), 2)) >= pow(in_n[0], 2)) {
		std::cout << "Cutting off invalid waves with imagnery sz..." << std::endl;
		std::cout << "[WARNING] " << "Propagation directions for decomposed waves are not all downward. We suggest to increase in_size or decrease the wavelength to tolerate higher Fourier coefficients. Constraints: (float(M[0]/2)/size[2])^2 + (float(M[1]/2)/size[1])^2 < (n/lambda)^2" << std::endl;
		if (M[0] != 1 && M[1] != 1) {
			if ((double(M[0] / 2) / in_size[0] * in_lambda + dir[0]) > (std::complex<double>(0.71, 0) * in_n[0]).real()) {
				M[0] = int(0.707 * in_n[0] * 2 * in_size[0] / in_lambda) - 1;
				std::cout << "M[0] is corrected as " << M[0] << std::endl;
			}
			if ((double(M[1] / 2) / in_size[1] * in_lambda + dir[1]) > (std::complex<double>(0.71, 0) * in_n[0]).real()) {
				M[1] = int(0.707 * in_n[1] * 2 * in_size[1] / in_lambda) - 1;
				std::cout << "M[1] is corrected as " << M[1] << std::endl;
			}
		}
		if (M[0] == 1) {
			if ((double(M[1] / 2) / in_size[1] * in_lambda + abs(dir[1])) >= in_n[0]) {
				M[1] = int(in_n[1] * 2 * in_size[1] / in_lambda - pow(10, -10));
				if (M[1] % 2 == 0)
					M[1] -= 1;
				std::cout << "For the 1-d volume, M[1] is corrected as " << M[1] << std::endl;
			}
		}
		if (M[1] == 1) {
			if ((double(M[0] / 2) / in_size[1] * in_lambda + abs(dir[0])) >= in_n[0]) {
				M[0] = int(in_n[0] * 2 * in_size[0] / in_lambda - pow(10, -10));
				if (M[0] % 2 == 0)
					M[0] -= 1;
				std::cout << "For the 1-d volume, M[0] is corrected as " << M[0] << std::endl;
			}
		}
	}
	MF = M[0] * M[1];

	std::chrono::time_point<std::chrono::system_clock> D_before = std::chrono::system_clock::now();
	D = Volume.CalculateD(M, dir);	// Calculate the property matrix for the sample
	std::chrono::time_point<std::chrono::system_clock> D_after = std::chrono::system_clock::now();
	elapsed_seconds = D_after - D_before;
	proffile << "Time for property matrix D: " << elapsed_seconds.count() << " s. " << std::endl << std::endl;

	// Fourier transform for the incident waves
	E0.push_back(e[0]);
	E0.push_back(e[1]);
	E0.push_back(e[2]);
	std::vector<Eigen::MatrixXcd> Ef(3);
	Ef[0] = fftw_fft2(E0[0] * Eigen::MatrixXcd::Ones(num_pixels[1], num_pixels[2]), M[1], M[0]);	// M[0]=3 is column. M[1]=1 is row. 
	Ef[1] = fftw_fft2(E0[1] * Eigen::MatrixXcd::Ones(num_pixels[1], num_pixels[2]), M[1], M[0]);
	Ef[2] = fftw_fft2(E0[2] * Eigen::MatrixXcd::Ones(num_pixels[1], num_pixels[2]), M[1], M[0]);
	EF.resize(3 * MF);
	EF.segment(0, MF) = Eigen::Map<Eigen::VectorXcd>(Ef[0].data(), MF);
	EF.segment(MF, MF) = Eigen::Map<Eigen::VectorXcd>(Ef[1].data(), MF);
	EF.segment(2 * MF, MF) = Eigen::Map<Eigen::VectorXcd>(Ef[2].data(), MF);

	// Sync the Fourier transform of direction propagation with Volume
	Sx = Eigen::Map<Eigen::RowVectorXcd>(Volume._meshS0.data(), MF);
	Sy = Eigen::Map<Eigen::RowVectorXcd>(Volume._meshS1.data(), MF);
	Sz[0] = Eigen::Map<Eigen::RowVectorXcd>(Volume._Sz[0].data(), MF);
	Sz[1] = Eigen::Map<Eigen::RowVectorXcd>(Volume._Sz[1].data(), MF);

	//std::cout << "Sx: " << Sx << std::endl;
	//std::cout << "sy: " << Sy << std::endl;
	//std::cout << "Sz[0]: " << Sz[0] << std::endl;
	//std::cout << "Sz[1]: " << Sz[1] << std::endl;

	if (logfile) {
		logfile << "Ex fourier form:" << EF.segment(0, MF) << std::endl;
		logfile << "Ey fourier form:" << EF.segment(MF, MF) << std::endl;
		logfile << "Ez fourier form:" << EF.segment(2 * MF, MF) << std::endl;
		logfile << "Sx fourier form:" << Sx << std::endl;
		logfile << "Sy fourier form:" << Sy << std::endl;
		logfile << "Sz0 fourier form:" << Sz[0] << std::endl;
		logfile << "Sz1 fourier form:" << Sz[1] << std::endl;
	}

	proffile << "Initialization finished." << std::endl;
	std::chrono::time_point<std::chrono::system_clock> initialized = std::chrono::system_clock::now();
	elapsed_seconds = initialized - start;
	proffile << "Time for initialization: " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	proffile << "Linear system starts..." << std::endl;
	// Build linear system
	InitMatrices();
	std::chrono::time_point<std::chrono::system_clock> initDone = std::chrono::system_clock::now();
	elapsed_seconds = initDone - initialized;
	proffile << "		Time for InitMatrices(): " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	SetGaussianConstraints();
	std::chrono::time_point<std::chrono::system_clock> gauss = std::chrono::system_clock::now();
	elapsed_seconds = gauss - initDone;
	proffile << "		Time for SetGaussianConstraints(): " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	SetBoundaryConditions();
	std::chrono::time_point<std::chrono::system_clock> boundary = std::chrono::system_clock::now();
	elapsed_seconds = boundary - gauss;
	proffile << "		Time for SetBoundaryConditions(): " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	proffile << "Linear system built." << std::endl;
	std::chrono::time_point<std::chrono::system_clock> built = std::chrono::system_clock::now();
	elapsed_seconds = built - initialized;
	proffile << "Time for building the system: " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	// MKL solution
	proffile << "Linear system solving (MKL)..." << std::endl;

	//const std::vector<long unsigned> shape{ (unsigned long)6 * MF, (unsigned long)6 * MF };
	//const bool fortran_order{ false };
	//npy::SaveArrayAsNumpy("A.npy", fortran_order, shape.size(), shape.data(), &A(0, 0));

	MKL_linearsolve(A, b);
	Eigen::VectorXcd x = b;
	//std::cout << "x: " << x << std::endl;
	proffile << "Linear system solved." << std::endl;

	std::chrono::time_point<std::chrono::system_clock> solved = std::chrono::system_clock::now();
	elapsed_seconds = solved - built;
	proffile << "Time for solving the system: " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	proffile << "Field reorganization starts..." << std::endl;
	if (logfile) {
		logfile << "x = " << x << std::endl;
		logfile << "Linear system solved." << std::endl << std::endl << std::endl;
		logfile << "Field reorganization starts..." << std::endl;
	}

	// The data structure that all data goes to
	CoupledWaveStructure<double> cw;
	cw.Layers.resize(L);
	size_t MF4 = MF * 4;																			// M is the length of beta/gamma/gg

	// store the incident plane wave
	int ind = (M[1] / 2) * M[0] + (M[0] / 2);
	tira::planewave<double> i(Sx(ind) * k, Sy(ind) * k , Sz[0](ind) * k* in_n[0], EF(ind), EF(MF + ind), EF(2 * MF + ind));
	cw.Pi.push_back(i);

	tira::planewave<double> zeros(0, 0, k, 0, 0, 0);
	for (int kk = 0; kk < M[1]; kk++) {
		for (int j = 0; j < M[0]; j++) {
			std::complex<double> sy = (kk - M[1] / 2) / in_size[1] * in_lambda + dir[1];
			std::complex<double> sx = (j - M[0] / 2) / in_size[0] * in_lambda + dir[0];
			if (true) {
				//if (abs((sx * sx + sy * sy).real()) < in_n[0] * in_n[0]) {
				counts += 1;
				int p = kk * M[0] + j;
				std::vector<tira::planewave<double>> P = mat2waves(i, x, p);
				// generate plane waves from the solution vector
				tira::planewave<double> r, t;
				for (size_t l = 0; l < L; l++) {														// for each layer
					if (l == 0) {
						cw.Layers[l].z = z[l];
						r = P[1 + l * 2 + 0].wind(0.0, 0.0, -z[l]);
						//r = P[1 + l * 2 + 0];
						cw.Layers[l].Pr.push_back(r);
					}
					if (l == L - 1) {
						cw.Layers[l].z = z[l];
						//t = P[1 + (l - 1) * 2 + 1];
						t = P[1 + (l - 1) * 2 + 1].wind(0.0, 0.0, -z[l]);
						cw.Layers[l].Pt.push_back(t);
					}
					if (logfile) {
						logfile << "LAYER " << l << "==========================" << std::endl;
						logfile << "i (" << p << ") ------------" << std::endl << i.str() << std::endl;
						logfile << "r (" << p << ") ------------" << std::endl << r.str() << std::endl;
						logfile << "t (" << p << ") ------------" << std::endl << t.str() << std::endl;
						logfile << std::endl;
					}
				}
			}
		}
	}
	if (counts != M[0] * M[1]) {
		saveSampleTexture = false;
		std::cout << "[WARNING] Not all decomposed Fourier waves are valid. We suggest to increase in_size or decrease the wavelength to tolerate higher Fourier coefficients. Constraints: (float(M[0]/2)/size[2])^2 + (float(M[1]/2)/size[1])^2 < (n/lambda)^2. Scatterviewsample is auto disabled. Please use Scatterview instead." << std::endl;
	}
	cw.isHete = saveSampleTexture;
	// Calculate beta according to the GD, GC, and Pt/Pr
	if (cw.isHete) {
		cw.M[0] = M[0];
		cw.M[1] = M[1];
		cw.size[0] = in_size[0];
		cw.size[1] = in_size[1];
		cw.size[2] = in_size[2];
		cw.Slices.resize(num_pixels[0]);
		cw.NIf.resize(num_pixels[0]);
		for (size_t i = 0; i < num_pixels[0]; i++) {
			cw.NIf[i].resize(M[0] * M[1]);
			for (int j = 0; j < M[0] * M[1]; j++) {
				cw.NIf[i][j] = Volume.NIf[i](j);
			}
		}
		std::chrono::time_point<std::chrono::system_clock> beta_before = std::chrono::system_clock::now();

		Eigen::MatrixXcd EF_mat;
		Eigen::MatrixXcd Pr_0;
		Eigen::MatrixXcd beta;

		EF_mat = Eigen::Map< Eigen::MatrixXcd>(EF.data(), 3 * MF, 1);
		Pr_0 = Eigen::Map< Eigen::MatrixXcd>(x.data(), 3 * MF, 1);

		for (size_t i = 0; i < num_pixels[0]; i++) {
			if (i == 0) {
				tmp = MKL_inverse(GD[i]);
				tmp_2 = MKL_multiply(tmp, f1, 1);
				beta = MKL_multiply(tmp_2, EF_mat, 1);

				tmp_2 = MKL_multiply(tmp, f2, 1);
				beta += MKL_multiply(tmp_2, Pr_0, 1);
			}
			else {
				tmp = MKL_inverse(GD[i]);
				tmp_2 = MKL_multiply(tmp, GC[i - 1], 1);
				tmp = MKL_multiply(tmp_2, beta, 1);
				beta = tmp;
			}
			Beta[i] = beta;
		}
		std::chrono::time_point<std::chrono::system_clock> beta_end = std::chrono::system_clock::now();
		elapsed_seconds = beta_end - beta_before;
		proffile << "Solving beta takes:" << elapsed_seconds.count() << "s" << std::endl;
		for (size_t i = 0; i < num_pixels[0]; i++) {
			cw.Slices[i].beta.resize(MF4);
			cw.Slices[i].gamma.resize(MF4);
			cw.Slices[i].gg.resize(MF4 * MF4);
			for (size_t m = 0; m < MF4; m++) {
				cw.Slices[i].beta[m] = Beta[i](m);
				cw.Slices[i].gamma[m] = eigenvalues[i](m);
			}
			for (size_t m = 0; m < MF4 * MF4; m++) {
				cw.Slices[i].gg[m] = eigenvectors[i](m);
			}
		}
	}
	std::cout << "Field saved in " << in_outfile << "." << std::endl;
	std::cout << "Number of pixels (x, y): [" << num_pixels[2] << "," << num_pixels[1] << "]" << std::endl;
	std::cout << "Number of sublayers: " << num_pixels[0] << std::endl;
	std::cout << "Number of Fourier coefficients (Mx, My): [" << M[0] << "," << M[1] << "]" << std::endl;
	std::cout << "Effective number of Fourier coefficients: [" << counts << "]" << std::endl;

	std::chrono::time_point<std::chrono::system_clock> save_before = std::chrono::system_clock::now();
	if (in_outfile != "") {
		cw.save(in_outfile);
	}
	std::chrono::time_point<std::chrono::system_clock> save_end = std::chrono::system_clock::now();
	elapsed_seconds = save_end - save_before;
	proffile << "Time for saving the field " << elapsed_seconds.count() << "s" << std::endl << std::endl << std::endl;
	elapsed_seconds = save_end - start;
	proffile << "Total time:" << elapsed_seconds.count() << "s" << std::endl;
}
