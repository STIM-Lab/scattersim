#include <iostream>
#include <tira/optics/planewave.h>
#include "CoupledWaveStructure.h"
#include "FourierWave.h"
#include <complex>
#include <math.h>
#include <fstream>
#include <boost/program_options.hpp>
#include <random>
#include <iomanip>
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "Eigen/Dense"
#include "cnpy/cnpy.h"
#include<time.h>

std::vector<double> in_dir;
double in_lambda;

std::vector<double> in_n;
std::vector<double> in_kappa;
std::vector<double> in_ex;
std::vector<double> in_ey;
std::vector<double> in_z;
std::vector<double> in_pos_interest;
std::vector<size_t> in_num_pixels;
std::vector<double> in_normal;
double in_na;
std::string in_outfile;
double in_alpha;
double in_beta;
std::string in_sample;
std::string in_mode;
std::vector<double> in_z_obj;
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
std::complex<double>* sz;
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
std::vector<Eigen::MatrixXcd> D;		// The property matrix
std::vector<int> fz;					// The coordinates for the sample boundaries
Eigen::VectorXcd eigenvalues;			// eigen values for current layer
Eigen::MatrixXcd eigenvectors;			// eigen vectors for current layer
Eigen::MatrixXcd Gc;					// Upward
Eigen::MatrixXcd Gd;					// Downward
std::vector<Eigen::VectorXcd > Eigenvalues;			// Dimension: (layers, coeffs)
std::vector<Eigen::MatrixXcd> Eigenvectors;			// Dimension: (layers, coeffs)
Eigen::MatrixXcd GD;					// Dimension: (layers, coeffs)
Eigen::MatrixXcd GC;					// Dimension: (layers, coeffs)
Eigen::MatrixXcd f1;
Eigen::MatrixXcd f2;
Eigen::MatrixXcd f3;
std::vector<double> z_new(2);

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
	for (size_t l = 0; l < L; l++)
		z[l] = in_z[l];
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
/// Sort the eigenvalues and eigenvectors by pairs
/// </summary>
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
	
	sort(eiV.begin(), eiV.end(), &sorter);

	eigenvalues.resize(len);
	eigenvectors.resize(len, len);

	if (logfile) {
		logfile << "eigenvalues_unordered: " << std::endl;
		logfile << eigenvalues_unordered << std::endl << std::endl;
		logfile << "eigenvectors_unordered: " << std::endl;
		logfile << eigenvectors_unordered << std::endl << std::endl;
	}
	for (size_t i = 0; i < len / 2; i++) {
		eigenvalues[2 * i] = eigenvalues_unordered[len - 1 - eiV[i].idx];
		eigenvectors.col(2 * i) = eigenvectors_unordered.col(len - 1 - eiV[i].idx);
		eigenvalues[2 * i + 1] = eigenvalues_unordered[eiV[i].idx];
		eigenvectors.col(2 * i + 1) = eigenvectors_unordered.col(eiV[i].idx);
	}
	if (logfile) {
		logfile << "eigenvalues: " << std::endl;
		logfile << eigenvalues << std::endl << std::endl;
		logfile << "eigenvectors: " << std::endl;
		logfile << eigenvectors << std::endl << std::endl;
	}

	Eigenvalues.push_back(eigenvalues);				// For computing the inner structure of the sample
	Eigenvectors.push_back(eigenvectors);
}

// Do eigen decomposition for Phi. 
// Sort the eigenvectors and eigenvalues by pairs. 
// Build matrices Gd and Gc.
void EigenDecompositionD() {
	for (size_t j = 0; j < D.size(); j++) {

		std::cout << "				Eigen solver working..." << std::endl;
		clock_t essolver1 = clock();
		Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(D[j]);
		Eigen::VectorXcd eigenvalues_unordered = es.eigenvalues();
		Eigen::MatrixXcd eigenvectors_unordered = es.eigenvectors();
		clock_t essolver2 = clock();
		std::cout << "				Time for eigen solver: " << (essolver2 - essolver1) / CLOCKS_PER_SEC << "s" << std::endl;

		//Eigen_Sort(eigenvalues_unordered, eigenvectors_unordered);		// Sort the eigenvalues

		eigenvalues = eigenvalues_unordered;
		eigenvectors = eigenvectors_unordered;
		Gd.resize(4 * MF, 4 * MF);
		Gc.resize(4 * MF, 4 * MF);
		for (size_t i = 0; i < eigenvalues.size(); i++) {
			z_new[0] = (double)in_pos_interest[4] + (double)fz[j] * (double)(in_pos_interest[5] - in_pos_interest[4]) / (double)in_num_pixels[2];
			z_new[1] = (double)in_pos_interest[4] + (double)fz[j + 1] * (double)(in_pos_interest[5] - in_pos_interest[4]) / (double)in_num_pixels[2];
			std::complex<double> Di = std::exp(std::complex<double>(0, 1) * k * eigenvalues(i) * (std::complex<double>)(z_new[1] - z_new[0]));
			std::complex<double> Ci = std::exp(std::complex<double>(0, 1) * k * eigenvalues(i) * (std::complex<double>)(z_new[0] - z_new[1]));

			if (i % 2 == 0) {
				Gd.col(i) = eigenvectors.col(i) * Di;
				Gc.col(i) = eigenvectors.col(i);
			}

			else {
				Gd.col(i) = eigenvectors.col(i);
				Gc.col(i) = eigenvectors.col(i) * Ci;
			}
		}
		if (j == 0) {
			GD = Gd;
			GC = Gc;
		}
		else
			GC = Gc * Gd.inverse() * GC;
		if (logfile) {
			logfile << "----------For the " << j << "th layer---------- " << std::endl;
			logfile << "Property matrix D: " << std::endl;
			logfile << D[j] << std::endl;
			logfile << "GD: " << std::endl;
			logfile << GD << std::endl;
			logfile << "GD inversese: " << std::endl;
			logfile << Gd.inverse() << std::endl;			
			logfile << "GC: " << std::endl;
			logfile << GC << std::endl;
			logfile << "GC inversese: " << std::endl;
			logfile << GC.inverse() << std::endl;
		}
	}
}

void MatTransfer(){
	f1.resize(4 * MF, 3 * MF);
	f2.resize(4 * MF, 3 * MF);
	f3.resize(4 * MF, 3 * MF);
	f1.setZero();
	f2.setZero();
	f3.setZero();

	Eigen::RowVectorXcd phase = (std::complex<double>(0, 1) * k * (std::complex<double>)(z[0] - z[0]) * Eigen::Map<Eigen::RowVectorXcd>(Sz[0].data(), Sz[0].size())).array().exp();
	Eigen::MatrixXcd Phase = phase.replicate(MF, 1);		// Phase is the duplicated (by row) matrix from phase.

	Eigen::MatrixXcd SZ0 = Sz[0].replicate(MF, 1);		// neg_SZ0 is the duplicated (by row) matrix from neg_Sz0.
	Eigen::MatrixXcd SZ1 = Sz[1].replicate(MF, 1);		// neg_SZ0 is the duplicated (by row) matrix from neg_Sz0.
	Eigen::MatrixXcd SX= Sx.replicate(MF, 1);		// neg_SX is the duplicated (by row) matrix from phase.
	Eigen::MatrixXcd SY = Sy.replicate(MF, 1);		// neg_SX is the duplicated (by row) matrix from phase.

	Eigen::MatrixXcd identity = Eigen::MatrixXcd::Identity(MF, MF);
	
	// first constraint (Equation 8)
	f1.block(0, 0, MF, MF) = identity.array() * Phase.array();
	f1.block(MF, MF, MF, MF) = identity.array() * Phase.array();
	f1.block(2 * MF, MF, MF, MF) = (std::complex<double>(-1, 0)) * identity.array() * Phase.array() * SZ0.array();
	f1.block(2 * MF, 2 * MF, MF, MF) =  identity.array() * Phase.array() * SY.array();
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

		logfile << "Special recording 'Gd * Gc.inv() * f3': " << std::endl;
		logfile << Gd * Gc.inverse() * f3 << std::endl;
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
	std::cout << "		Eigen decomposition starts..." << std::endl;
	clock_t eigen1 = clock();
	EigenDecompositionD();		// Compute Gd and Gc
	clock_t eigen2 = clock();
	std::cout << "		Time for eigen decomposition: " << (eigen2 - eigen1) / CLOCKS_PER_SEC << "s" << std::endl;
	MatTransfer();				// Achieve the connection between the variable vector and the field vector

	A.block(2 * MF, 0, 4 * MF, 3 * MF) = f2;
	A.block(2 * MF, 3 * MF, 4 * MF, 3 * MF) = Gd * Gc.inverse() * f3;

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
	glm::vec<3, double> s = i.getDirection();
	tira::planewave<double> r(Sx(p) * k,
		Sy(p) * k,
		-Sz[0](p) * k,
		x[idx(0, Reflected, X, p, MF)],
		x[idx(0, Reflected, Y, p, MF)],
		x[idx(0, Reflected, Z, p, MF)]);

	tira::planewave<double> t(Sx(p) * k,
		Sy(p) * k,
		Sz[1](p) * k,
		x[idx(1, Transmitted, X, p, MF)],
		x[idx(1, Transmitted, Y, p, MF)],
		x[idx(1, Transmitted, Z, p, MF)]);

	//std::cout << "x: " << x << std::endl;
	//std::cout << "id for r: " << idx(0, Reflected, X, p, MF) << std::endl;
	//std::cout << "id for r: " << idx(0, Reflected, Y, p, MF) << std::endl;
	//std::cout << "id for r: " << idx(0, Reflected, Z, p, MF) << std::endl;
	//std::cout << "Sx(p) * k for r: " << Sx(p) * k << std::endl;
	//std::cout << "Sy(p) * k for r: " << Sy(p) * k << std::endl;
	//std::cout << "-Sz[0](p) * k for r: " << -Sz[0](p) * k << std::endl;
	//std::cout << "x for r: " << x[idx(0, Reflected, X, p, MF)] << std::endl;
	//std::cout << "x for r: " << x[idx(0, Reflected, Y, p, MF)] << std::endl;
	//std::cout << "x for r: " << x[idx(0, Reflected, Z, p, MF)] << std::endl;

	//std::cout << "id for t: " << idx(1, Transmitted, X, p, MF) << std::endl;
	//std::cout << "id for t: " << idx(1, Transmitted, Y, p, MF) << std::endl;
	//std::cout << "id for t: " << idx(1, Transmitted, Z, p, MF) << std::endl;
	//std::cout << "Sx(p) * k for r: " << Sx(p) * k << std::endl;
	//std::cout << "Sy(p) * k for r: " << Sy(p) * k << std::endl;
	//std::cout << "Sz[1](p) * k for r: " << Sz[1](p) * k << std::endl;
	//std::cout << "x for t: " << x[idx(1, Transmitted, X, p, MF)] << std::endl;
	//std::cout << "x for t: " << x[idx(1, Transmitted, Y, p, MF)] << std::endl;
	//std::cout << "x for t: " << x[idx(1, Transmitted, Z, p, MF)] << std::endl;

	P.push_back(r);
	P.push_back(t);
	return P;
}

/// Removes waves in the input set that have a k-vector pointed along the negative z axis
std::vector< tira::planewave<double> > RemoveInvalidWaves(std::vector<tira::planewave<double>> W){
	std::vector<tira::planewave<double>> new_W;
	for(size_t i = 0; i < W.size(); i++){
		if(W[i].getKreal()[2] >0)
			new_W.push_back(W[i]);
	}

	return new_W;
}


int main(int argc, char** argv) {
	clock_t start = clock();
	std::cout << "Initialization starts..." << std::endl;

	// Set up all of the input options provided to the user
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("lambda", boost::program_options::value<double>(&in_lambda)->default_value(1.0), "incident field vacuum wavelength")
		("direction", boost::program_options::value<std::vector<double> >(&in_dir)->multitoken()->default_value(std::vector<double>{0, 0, 1}, "0, 0, 1"), "incoming field direction")
		("ex", boost::program_options::value<std::vector<double> >(&in_ex)->multitoken()->default_value(std::vector<double>{0, 0}, "0, 0"), "incoming field direction")
		("ey", boost::program_options::value<std::vector<double> >(&in_ey)->multitoken()->default_value(std::vector<double>{1, 0}, "1, 0"), "incoming field direction")
		("n", boost::program_options::value<std::vector<double>>(&in_n)->multitoken()->default_value(std::vector<double>{1.0, 1.0}, "1, 1"), "real refractive index (optical path length) of the upper and lower layers")
		("kappa", boost::program_options::value<std::vector<double> >(&in_kappa)->multitoken()->default_value(std::vector<double>{0}, "0.00"), "absorbance of the lower layer (upper layer is always 0.0)")
		("n_sample", boost::program_options::value<double>(&in_n_sample)->multitoken()->default_value(1.4, "1.4"), "real refractive index (optical path length) of the sample")
		("kappa_sample", boost::program_options::value<double>(&in_kappa_sample)->multitoken()->default_value(0.07, "0.07"), "absorbance of the sample")
		("z", boost::program_options::value<std::vector<double> >(&in_z)->multitoken()->default_value(std::vector<double>{-3.0, 3.0}, "-3.0, 3.0"), "position of the sample layer boundary")
		("output", boost::program_options::value<std::string>(&in_outfile)->default_value("py.cw"), "output filename for the coupled wave structure")
		("alpha", boost::program_options::value<double>(&in_alpha)->default_value(1), "angle used to focus the incident field")
		("beta", boost::program_options::value<double>(&in_beta)->default_value(0.0), "internal obscuration angle (for simulating reflective optics)")
		("na", boost::program_options::value<double>(&in_na), "focus angle expressed as a numerical aperture (overrides --alpha)")
		("coefficients", boost::program_options::value<std::vector<int> >(&in_coeff)->multitoken()->default_value(std::vector<int>{201, 1}, "3, 3"), "number of Fouerier coefficients (can be specified in 2 dimensions)")
		("mode", boost::program_options::value<std::string>(&in_mode)->default_value("polar"), "sampling mode (polar, montecarlo)")
		("log", "produce a log file")
		// input just for scattervolume 
		("center", boost::program_options::value<std::vector<double>>(&in_center)->multitoken()->default_value(std::vector<double>{0, 0, 0}, "0, 0, 0"), "The coordinates of the sample center")
		("pos_interest", boost::program_options::value<std::vector<double>>(&in_pos_interest)->multitoken()->default_value(std::vector<double>{-100, 100, -100, 100, -100, 100}, "-100, 100, -100, 100, -100, 100"), "The region of the whole field")
		("rec_bar", boost::program_options::value<std::vector<double> >(&in_rec_bar)->multitoken()->default_value(std::vector<double>{ 50 }, "50"), "All information about the sample: width=50")
		("circle", boost::program_options::value<std::vector<double> >(&in_circle)->multitoken()->default_value(std::vector<double>{ 0, 0, -1.5, 3, 1.96 }, "0 0 -1.5 3 1.96"), "position of the shape center, diameter, and its refractive index")
		("num_pixels", boost::program_options::value<std::vector<size_t> >(&in_num_pixels)->multitoken()->default_value(std::vector<size_t>{200, 200, 200}, "200, 2, 100"), "number of pixels along y, x and z axes")
		("sample", boost::program_options::value<std::string>(&in_sample)->default_value("rec_bar"), "Sample type: rec_bar, rec, grating, circle, sphere")
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

											
	if(vm.count("log")){									// if a log is requested, begin output
		std::stringstream ss;
		ss<<std::time(0)<<"_scattervolume.log";
		logfile.open(ss.str());
	}

	// override alpha with NA if specified
	if (vm.count("na")) {
		in_alpha = asin(in_na);
	}

	// Calculate the number of layers based on input parameters (take the maximum of all layer-specific command-line options)
	L = in_n.size();
	if (in_kappa.size() + 1 > L)																				// if more absorption coefficients are provided
		L = in_kappa.size() + 1;																				// add additional layers
	if (in_z.size() > L)																					// if more z coordinates are provided
		L = in_z.size() + 1;																					// add additional layers

	// update parameter lists so that all represent the same number of layers
	in_n.resize(L, in_n.back());																				// add additional layers (append copies of the previous refractive index)
	in_kappa.resize(L - 1, 0.0);																				// add additional layers (append values of zero absorbance as necessary)
	if (in_z.size() + 1 < L) {																					// if there isn't a z coordinate specified for each layer
		for (size_t l = in_z.size(); l < L - 1; l++) {
			in_z.push_back(in_z.back() + 10.0);																	// add additional layers in increments of 10 units
		}
	}

	// Get the number of the Fourier Coefficients
	if (in_coeff.size() == 1) {
		M[0] = std::sqrt(in_coeff[0]);
		M[1] = std::sqrt(in_coeff[0]);
	}
	else if (in_coeff.size() == 2) {
		M[0] = in_coeff[0];
		M[1] = in_coeff[1];
	}
	else {
		M[0] = 1;
		M[1] = 1;
	}
	MF = M[0] * M[1];

	Eigen::Vector3d dir(in_dir[0], in_dir[1], in_dir[2]);
	dir.normalize();																							// set the normalized direction of the incoming source field

	// wavenumber
	k = (std::complex<double>)(2 * M_PI / in_lambda * in_n[0]);

	// store all of the layer positions and refractive indices
	InitLayerProperties();
	// Define sample volume, reformat, and reorgnize.
	volume < std::complex< double> > Volume("RecBar.npy", ni, z, in_center, in_rec_bar[0], in_pos_interest, k.real(), std::complex<double>(in_n_sample, in_kappa_sample));
	in_num_pixels = Volume.reformat();
	fz = Volume.reorg();				// Form fz and flag
	D = Volume.CalculateD(M, dir);	// Calculate the property matrix for the sample

	// Fourier transform for the incident waves
	E0.push_back(std::complex<double>(in_ex[0], in_ex[1]));
	E0.push_back(std::complex<double>(in_ey[0], in_ey[1]));
	E0.push_back(std::sqrt(pow(std::complex<double>(1, 0), 2) - pow(E0[0], 2) - pow(E0[1], 2)));	
	std::vector<Eigen::MatrixXcd> Ef(3);
	Ef[0] = fftw_fft2(E0[0] * Eigen::MatrixXcd::Ones(in_num_pixels[0], in_num_pixels[1]), M[1], M[0]);	// M[0]=3 is column. M[1]=1 is row. 
	Ef[1] = fftw_fft2(E0[1] * Eigen::MatrixXcd::Ones(in_num_pixels[0], in_num_pixels[1]), M[1], M[0]);
	Ef[2] = fftw_fft2(E0[2] * Eigen::MatrixXcd::Ones(in_num_pixels[0], in_num_pixels[1]), M[1], M[0]);
	EF.resize(3 * MF);
	EF.segment(0, MF) = Eigen::Map<Eigen::VectorXcd>(Ef[0].data(), MF);
	EF.segment(MF, MF) = Eigen::Map<Eigen::VectorXcd>(Ef[1].data(), MF);
	EF.segment(2 * MF, MF) = Eigen::Map<Eigen::VectorXcd>(Ef[2].data(), MF);

	// Sync the Fourier transform of direction propagation with Volume
	Sx = Eigen::Map<Eigen::RowVectorXcd>(Volume._meshS0.data(), MF);
	Sy = Eigen::Map<Eigen::RowVectorXcd>(Volume._meshS1.data(), MF);
	Sz[0] = Eigen::Map<Eigen::RowVectorXcd>(Volume._Sz[0].data(), MF);
	Sz[1] = Eigen::Map<Eigen::RowVectorXcd>(Volume._Sz[1].data(), MF);

	if (logfile) {
		logfile << "Ex fourier form:" << EF.segment(0, MF) << std::endl;
		logfile << "Ey fourier form:" << EF.segment(MF, MF) << std::endl;
		logfile << "Ez fourier form:" << EF.segment(2 * MF, MF) << std::endl;
		logfile << "Sx fourier form:" << Sx << std::endl;
		logfile << "Sy fourier form:" << Sy << std::endl;
		logfile << "Sz0 fourier form:" << Sz[0] << std::endl;
		logfile << "Sz1 fourier form:" << Sz[1] << std::endl;
	}

	std::cout << "Initialization finished." << std::endl;
	clock_t initialized = clock();
	std::cout << "Time for initialization: " << (initialized - start) / CLOCKS_PER_SEC << "s" << std::endl << std::endl;

	std::cout << "Linear system starts..." << std::endl ;
	// Build linear system
	InitMatrices();
	SetGaussianConstraints();
	SetBoundaryConditions();

	std::cout << "Linear system built." << std::endl;
	clock_t built = clock();
	std::cout << "Time for building the system: " << (built - initialized) / CLOCKS_PER_SEC << "s" << std::endl << std::endl;

	std::cout << "Linear system solving..." << std::endl;
	Eigen::VectorXcd x = A.colPivHouseholderQr().solve(b);		
	std::cout << "Linear system solved." << std::endl;
	clock_t solved = clock();
	std::cout << "Time for solving the system: " << (solved - built) / CLOCKS_PER_SEC << "s" << std::endl << std::endl;

	std::cout << "Field reorganization starts..." << std::endl;
	if (logfile) {
		logfile << "x = " << x << std::endl;
		logfile << "Linear system solved." << std::endl << std::endl << std::endl;
		logfile << "Field reorganization starts..." << std::endl;
	}

	// The data structure that all data goes to
	CoupledWaveStructure<double> cw;
	cw.Layers.resize(L);					

	for (size_t p = 0; p < MF; p++) {															// for each incident plane wave
		tira::planewave<double> zero(0, 0, 1, 0, 0);																		// store the incident plane wave in i
		tira::planewave<double> i(Sx(p) * k, Sy(p) * k, -Sz[0](p) * k, EF(p), EF(MF + p));																		// store the incident plane wave in i
		cw.Pi.push_back(i);

		std::vector<tira::planewave<double>> P = mat2waves(i, x, p);	

		// generate plane waves from the solution vector
		tira::planewave<double> r, t;
		for (size_t l = 0; l < L; l++) {														// for each layer
			if (l == 0) {
				cw.Layers[l].z = z_new[l];
				r = P[1 + l * 2 + 0].wind(0.0, 0.0, -z_new[l]);
				cw.Layers[l].Pr.push_back(r);
				t = zero;
				cw.Layers[l].Pt.push_back(t);
			}
			if (l == L - 1) {
				cw.Layers[l].z = z_new[l];
				r = zero;
				cw.Layers[l].Pr.push_back(r);
				t = P[1 + (l - 1) * 2 + 1].wind(0.0, 0.0, -z_new[l]);
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
	std::cout << "Field saved in " << in_outfile << "." << std::endl;
	clock_t simulated = clock();
	std::cout << "Time for saving the field " << (simulated - solved) / CLOCKS_PER_SEC << "s" << std::endl << std::endl << std::endl;

	std::cout << "Number of pixels (x, y, z): [" << in_num_pixels[1] << "," << in_num_pixels[0] << "," << in_num_pixels[2] << "]" << std::endl;
	std::cout << "Number of Fourier coefficients (Mx, My): [" << M[0] << "," << M[1] << "]" << std::endl;
	std::cout << "Total time:" << (simulated - start) / CLOCKS_PER_SEC << "s" << std::endl;

	if (in_outfile != "") {
		cw.save(in_outfile);
	}
}