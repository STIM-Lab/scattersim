#pragma once

#include "tira/optics/planewave.h"
#include "CoupledWaveStructure.h"
#include <vector>
#include <fstream>
#include "fftw3.h"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include <unsupported/Eigen/MatrixFunctions>
#include "tira/field.h"
#include "fftw3.h"

#define PI 3.141592653

// function to compute 2d-fftshift (like MATLAB) of a matrix
Eigen::MatrixXcd fftShift2d(Eigen::MatrixXcd mat)
{
	int m, n, p, q;
	m = mat.rows();
	n = mat.cols();

	// matrix to store fftshift data
	Eigen::MatrixXcd mat_fftshift(m, n);

	// odd # of rows and cols
	if ((int)fmod(m, 2) == 1)
	{
		p = (int)floor(m / 2.0);
		q = (int)floor(n / 2.0);
	}
	else // even # of rows and cols
	{
		p = (int)ceil(m / 2.0);
		q = (int)ceil(n / 2.0);
	}

	// vectors to store swap indices
	Eigen::RowVectorXi indx(m), indy(n);

	// compute swap indices
	if ((int)fmod(m, 2) == 1) // # of rows odd
	{
		for (int i = 0; i < m - p - 1; i++)
			indx(i) = (m - p) + i;
		for (int i = m - p - 1; i < m; i++)
			indx(i) = i - (m - p - 1);
	}
	else // # of rows even
	{
		for (int i = 0; i < m - p; i++)
			indx(i) = p + i;
		for (int i = m - p; i < m; i++)
			indx(i) = i - (m - p);
	}

	if ((int)fmod(n, 2) == 1) // # of cols odd
	{
		for (int i = 0; i < n - q - 1; i++)
			indy(i) = (n - q) + i;
		for (int i = n - q - 1; i < n; i++)
			indy(i) = i - (n - q - 1);
	}
	else // # of cols even
	{
		for (int i = 0; i < n - q; i++)
			indy(i) = q + i;
		for (int i = n - q; i < n; i++)
			indy(i) = i - (n - q);
	}

	// rearrange the matrix elements by swapping the elements
	// according to the indices computed above.
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			mat_fftshift(i, j) = mat(indx(i), indy(j));

	// return fftshift matrix
	return mat_fftshift;
}

/// </summary>
/// <param name="A">input image specified as Eiegn::MatrixXcd format</param>
/// <param name="M1">The kept number of Fourier coefficients along x</param>
/// <param name="M2">The kept number of Fourier coefficients along y</param>
/// <returns></returns>
Eigen::MatrixXcd fftw_fft2(Eigen::MatrixXcd A, int M1, int M2) {
	int N1 = A.rows();
	int N2 = A.cols();

	// Matrix to store Fourier domain data
	Eigen::MatrixXcd B(N1, N2);

	// For Fourier transform
	fftw_complex* in;
	fftw_complex* out;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N1 * N2);

	// convert a 2d-matrix into a 1d array
	for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
		{
			in[i * N2 + j][0] = A(i, j).real();
			in[i * N2 + j][1] = A(i, j).imag();
		}

	// allocate 1d array to store fft data
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N1 * N2);

	fftw_plan plan = fftw_plan_dft_2d(N1, N2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);

	// covert 1d fft data to a 2d-matrix
	for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
		{
			B(i, j) = std::complex<double>(out[i * N2 + j][0], out[i * N2 + j][1]);
		}

	// Scale the output
	B = B / N1 / N2;

	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);

	Eigen::MatrixXcd outnew;
	outnew = fftShift2d(B);
	return outnew.block(N1/2-M1/2, N2/2-M2/2, M1, M2);
}

/// <summary>
/// Function: Same as the numpy.meshgrid()
/// https://blog.csdn.net/weixin_41661099/article/details/105011027
/// </summary>
/// <param name="vecX"></param>
/// <param name="vecY"></param>
/// <param name="meshX"></param>
/// <param name="meshY"></param>
void meshgrid(Eigen::VectorXcd& vecX, Eigen::VectorXcd& vecY, Eigen::MatrixXcd& meshX, Eigen::MatrixXcd& meshY) {
	int vecXLength = vecX.size();
	int vecYLength = vecY.size();
	for (size_t i = 0; i < vecYLength; i++) {
		meshX.row(i) = vecX;
	}
	for (size_t i = 0; i < vecXLength; i++) {
		meshY.col(i) = vecY.transpose();
	}
}

template <class T>
class volume : public tira::field<T> {
public:
	std::vector<Eigen::MatrixXcd> _Sample;

	Eigen::VectorXcd _n_layers;
	double* _z = new double[2];
	std::vector<double> _center;
	double _width;
	std::vector<unsigned int> _num_pixels;
	std::vector<double> _pos_interest;
	std::complex<double> _n_volume;

	std::vector<int> _flag;
	std::vector<int> _fz;
	Eigen::VectorXd _Z;
	std::vector<Eigen::MatrixXcd> _Phi;
	int* _M = new int[2];
	double _k;

	Eigen::VectorXd _p_series;
	Eigen::VectorXd _q_series;
	Eigen::VectorXd _up;
	Eigen::VectorXd _wq;
	Eigen::VectorXcd _Sx;			// Fourier coefficients of x component of direction
	Eigen::VectorXcd _Sy;			// Fourier coefficients of y component of direction
	std::vector<Eigen::MatrixXcd> _Sz;		// 2D vector for the Fourier coefficients of z component for the upper and lower boundaries
	Eigen::MatrixXcd _meshS0, _meshS1;

	Eigen::VectorXd _dir;

	// For sparse storage
	std::vector<int> _M_rowInd;
	std::vector<int> _M_colInd;
	std::vector<std::complex<double>> _M_val;

	volume(std::string filename, Eigen::VectorXcd n_layers, double* z, std::vector<double> center, double width, std::vector<double> pos_interest, double k, std::complex<double> n_volume){
		// Read data from .npy file
		npy<std::complex<double>>(filename);

		// Necessary parameters
		_n_layers = n_layers;
		_z = z;
		_center = center;
		_width = width;
		_pos_interest = pos_interest;
		_k = k;
		_n_volume = n_volume;
	}

	/// <summary>
	/// Read data from .npy file as std::vector<double> and reformat it to be std::vector<Eigen::MatrixXcd>
	/// </summary>
	std::vector<size_t> reformat() {
		// _shape = [shape_x, shape_y, shape_z]
		_Sample.resize(_shape[2]);
		for (int i = 0; i < _Sample.size(); i++) {
			_Sample[i].resize(_shape[0], _shape[1]); // Orders for resize(): (row, col)
			_Sample[i] = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(&_data[i * _shape[0] * _shape[1]], _shape[0], _shape[1]);
			//if (i > 48 && i < 53)
			//	std::cout << "Sample[" << i << "]: " << _Sample[i] << std::endl;
		}
		return _shape;
	}

	/// <summary>
	/// Set labels for layers along z
	/// </summary>
	/// <returns></returns>
	std::vector<int> reorg() {
		_flag.resize(_shape[2]);
		int num = 0;
		int ind = 0;
		for (size_t i = 0; i < _shape[2]; i++) {
			// If _Z[i] is goes to a different layer
			if (i > 0 && !(_Sample[i].isApprox(_Sample[i - 1]))) {
				// If homogeneous
				if (_Sample[i].real().maxCoeff() == _Sample[i].real().minCoeff() && _Sample[i].imag().maxCoeff() == _Sample[i].imag().minCoeff()) {
					num += 1;
					_flag[i] = num;
				}
				// if heterogeneous
				else {
					ind -= 1;
					_fz.push_back(i);
					_flag[i] = ind;
					// if next layer is homogeneous, mark the current _z[i+1] as next fz.
					if (_Sample[i + 1].real().maxCoeff() == _Sample[i + 1].real().minCoeff() && _Sample[i + 1].imag().maxCoeff() == _Sample[i + 1].imag().minCoeff()) {
						_fz.push_back(i + 1);
						num += 1;
					}
				}
			}
			// If _Z[i] remains at the same layer
			else {
				// If homogeneous
				if (_Sample[i].real().maxCoeff() == _Sample[i].real().minCoeff() && _Sample[i].imag().maxCoeff() == _Sample[i].imag().minCoeff()) {
					_flag[i] = num;
				}
				else {
					_flag[i] = ind;
					if (_Sample[i + 1].real().maxCoeff() == _Sample[i + 1].real().minCoeff() && _Sample[i + 1].imag().maxCoeff() == _Sample[i + 1].imag().minCoeff()) {
						_fz.push_back(i + 1);
						num += 1;
					}
				}
			}
		}
		// For unit sample, manually set numbers for _fz
		if (_fz.size() == 0) {
			_fz.resize(2);
			double temp1 = abs(_pos_interest[4]);
			double temp2 = abs(_pos_interest[4]);
			double current;
			for (size_t i = 0; i < _shape[2]; i++) {
				current = _pos_interest[4] + (_pos_interest[5] - _pos_interest[4]) / (double)_shape[2] * (double)i;
				if (abs(current - _z[0]) < abs(temp1)) {
					temp1 = current;
					_fz[0] = i;
				}
				if (abs(current - _z[1]) < abs(temp2)) {
					temp2 = current;
					_fz[1] = i;
				}
			}
		}
		return _fz;
	}

	std::vector<Eigen::MatrixXcd> CalculateD(int* M, Eigen::VectorXd dir) {
		_M = M;
		_dir = dir;
		for (size_t i = 0; i < _fz.size() - 1; i++) {
			std::cout << "		For layer " << i << ", the property matrix D starts forming..." << std::endl;
			clock_t Phi1 = clock();
			_Phi.push_back(phi(_Sample[_fz[i]]));
			clock_t Phi2 = clock();
			std::cout << "		Time for forming D: " << (Phi2 - Phi1) / CLOCKS_PER_SEC << "s" << std::endl;
		}
		return _Phi;
	}

private:
	
	void UpWq_Cal() {
		_p_series.setLinSpaced(_M[0], -_M[0] / 2, (_M[0] - 1) / 2);
		_q_series.setLinSpaced(_M[1], -_M[1] / 2, (_M[1] - 1) / 2);
		_up = 2 * PI * _p_series / (_pos_interest[1] - _pos_interest[0]) + _dir[0] * _k * Eigen::VectorXd::Ones(_M[0]);
		_wq = 2 * PI * _q_series / (_pos_interest[1] - _pos_interest[0]) + _dir[1] * _k * Eigen::VectorXd::Ones(_M[1]);
		_Sx = (_up / _k).cast<std::complex<double>>();
		_Sy = (_wq / _k).cast<std::complex<double>>();

		_meshS0.setZero(_M[1], _M[0]);
		_meshS1.setZero(_M[1], _M[0]);
		// The z components for propagation direction. _Sz[0] is for the upper region while _Sz[1] is for the lower region
		meshgrid(_Sx, _Sy, _meshS0, _meshS1);
		_Sz.resize(2);
		_Sz[0] = (pow(_n_layers(0), 2) * Eigen::MatrixXcd::Ones(_M[1], _M[0]).array() - _meshS0.array().pow(2) - _meshS1.array().pow(2)).cwiseSqrt();
		_Sz[1] = (pow(_n_layers(1), 2) * Eigen::MatrixXcd::Ones(_M[1], _M[0]).array() - _meshS0.array().pow(2) - _meshS1.array().pow(2)).cwiseSqrt();
	}

	Eigen::MatrixXcd phi(Eigen::MatrixXcd sample) {
		_M_val.reserve(100000);
		_M_rowInd.reserve(100000);
		_M_colInd.reserve(100000);
		int idx = 0;
		UpWq_Cal();
		//std::cout <<"Dimension of the Property Matrix: ("<< sample.rows() << ", "<< sample.cols() << ")" << std::endl;
		Eigen::MatrixXcd Nf = fftw_fft2(sample.array().pow(2), _M[1], _M[0]);
		Eigen::MatrixXcd Nif = fftw_fft2(sample.cwiseInverse().array().pow(2), _M[1], _M[0]);
		int MF = _M[0] * _M[1];
		// Calculate Phi
		Eigen::MatrixXcd phi;
		phi.setZero(4 * MF, 4 * MF);

		std::vector<Eigen::VectorXcd> A(3, Eigen::VectorXcd(MF));
		double k_inv = 1.0 / pow(_k, 2);
		size_t indR, indC;
		for (size_t qi = 0; qi < _M[1]; qi++) {
			std::complex<double> wq = _wq[qi];
			for (size_t pi = 0; pi < _M[0]; pi++) {
				std::complex<double> up = _up[pi];
				size_t li = 0;
				for (size_t qj = 0; qj < _M[1]; qj++) {
					std::complex<double> wqj = _wq[qj];
					indR = ((int)_q_series[qi] % _M[1] - (int)_q_series[qj] % _M[1] + _M[1]) % _M[1];
					for (size_t pj = 0; pj < _M[0]; pj++) {
						std::complex<double> upj = _up[pj];
						indC = ((int)_p_series[pi] % _M[0] - (int)_p_series[pj] % _M[0] + _M[0]) % _M[0];			// % has different meanings in C++ and Python
						A[0](li) = Nf((indR + (_M[1] / 2)) % _M[1], (indC + (_M[0] / 2)) % _M[0]);
						A[1](li) = upj * (Nif((indR + (_M[1] / 2)) % _M[1], (indC + (_M[0] / 2)) % _M[0]));
						A[2](li) = wqj * (Nif((indR + (_M[1] / 2)) % _M[1], (indC + (_M[0] / 2)) % _M[0]));
						li += 1;
					}
				}

				// Dense storage
				phi.row(qi * _M[0] + pi).segment(2 * MF, MF) = up * k_inv * A[2];
				//std::cout << "sec 1 row 1: " << up * k_inv * A[2] << std::endl;
				phi.row(qi * _M[0] + pi).segment(3 * MF, MF) = -up * k_inv * A[1];
				//std::cout << "sec 1 row 2: " << -up * k_inv * A[1] << std::endl;
				phi(qi * _M[0] + pi, 3 * MF + qi * _M[0] + pi) += 1;
				//std::cout << "Index for sec 1 row 3: " << qi * _M[0] + pi << "," << 3 * MF + qi * _M[0] + pi << std::endl;

				phi.row(qi * _M[0] + pi + MF).segment(2 * MF, MF) = wq * k_inv * A[2];
				//std::cout << "sec 2 row 1: " << wq * k_inv * A[2] << std::endl;
				phi.row(qi * _M[0] + pi + MF).segment(3 * MF, MF) = -wq * k_inv * A[1];
				//std::cout << "sec 2 row 2: " << -wq * k_inv * A[1] << std::endl;
				phi(qi * _M[0] + pi + MF, 2 * MF + qi * _M[0] + pi) += -1;
				//std::cout << "Index for sec 2 row 3: " << qi * _M[0] + pi + MF << ", " << 2 * MF + qi * _M[0] + pi << std::endl;

				phi.row(qi * _M[0] + pi + 2 * MF).segment(MF, MF) = -A[0];
				//std::cout << "sec 3 row 1: " << -A[0] << std::endl;
				phi(qi * _M[0] + pi + 2 * MF, qi * _M[0] + pi) += -up * wq * k_inv;
				//std::cout << "sec 3 row 2: " << -up * wq * k_inv << std::endl;
				phi(qi * _M[0] + pi + 2 * MF, MF + qi * _M[0] + pi) += up * up * k_inv;
				//std::cout << "Index for sec 3 row 3: " << qi * _M[0] + pi + 2 * MF << ", " << MF + qi * _M[0] + pi << std::endl;
				//std::cout << "sec 3 row 3: " << up * up * k_inv << std::endl;

				phi.row(qi * _M[0] + pi + 3 * MF).segment(0, MF) = A[0];
				//std::cout << "sec 4 row 1: " << A[0] << std::endl;
				phi(qi * _M[0] + pi + 3 * MF, qi * _M[0] + pi) += -wq * wq * k_inv;
				//std::cout << "sec 4 row 2: " << -wq * wq * k_inv << std::endl;
				phi(qi * _M[0] + pi + 3 * MF, MF + qi * _M[0] + pi) += up * wq * k_inv;
				//std::cout << "Index for sec 4 row 3: " << qi * _M[0] + pi + 3 * MF << ", " << MF + qi * _M[0] + pi << std::endl;
				//std::cout << "sec 4 row 3: " << up * wq * k_inv << std::endl;
			}
		}
		for (size_t i = 0; i < phi.rows(); i++) {
			for (size_t j = 0; j < phi.cols(); j++) {
				_M_val.push_back(phi(i, j));
				_M_rowInd.push_back(i);
				_M_colInd.push_back(j);
			}
		}
		//std::cout << phi << std::endl;
		return phi;
	}
};
