#include <tira/optics/planewave.h>
#include <complex>
#include "Eigen/Eigen"
#include <iostream>
#include <extern/libnpy/npy.hpp>

#include "third_Lapack.h"

int main(int argc, char** argv) {
	Eigen::VectorXcd eigenvalues;
	Eigen::MatrixXcd eigenvectors;

	std::vector<std::complex<double>> loaded_data;
	std::vector<unsigned long> shape;
	bool is_fortran;
	npy::LoadArrayFromNumpy<std::complex<double>>("D.npy", shape, is_fortran, loaded_data);
	Eigen::Map<Eigen::MatrixXcd> D(loaded_data.data(), shape[0], shape[1]);
	int MF = shape[0] / 4;
	bool EIGEN = true;
	bool LAPACK = false;
	bool MKL_lapack = true;
	bool CUDA = false;

	if (EIGEN) {
		Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(D);
		eigenvalues = es.eigenvalues();
		eigenvectors = es.eigenvectors();
		std::cout << "eigenvalues from eigen solver: " << std::endl;
		std::cout << eigenvalues << std::endl;
		std::cout << "eigenvectors from eigen solver: " << std::endl;
		std::cout << eigenvectors << std::endl;
	}
	if (MKL_lapack) {
		std::complex<double>* A = new std::complex<double>[4 * MF * 4 * MF];
		Eigen::MatrixXcd::Map(A, D.rows(), D.cols()) = D;
		std::complex<double>* evl = new std::complex<double>[4 * MF];
		std::complex<double>* evt = new std::complex<double>[4 * MF * 4 * MF];
		clock_t s = clock();
		MKL_eigensolve(A, evl, evt, 4 * MF);
		clock_t e = clock();
		std::cout << "Time for MKL_eigensolve():" << (e - s) / CLOCKS_PER_SEC << "s" << std::endl;
		eigenvalues = Eigen::Map<Eigen::VectorXcd>(evl, 4 * MF);
		eigenvectors = Eigen::Map < Eigen::MatrixXcd, Eigen::ColMajor >(evt, 4 * MF, 4 * MF);

	}

	return 1;
}