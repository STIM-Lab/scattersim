#pragma once

//#include <thrust/copy.h>
//#include <thrust/device_vector.h>
//#include <thrust/fill.h>
//#include <thrust/host_vector.h>
//#include <thrust/sequence.h>


//#include "cusp/precond/diagonal.h"
//#include "cusp/blas/blas.h"
//#include "cusp/krylov/gmres.h"
//

//Eigen::VectorXcd CuspIterativeSolve(Eigen::MatrixXcd& A, Eigen::VectorXcd& b) {
//	//Transfer to sparse format
//
//	std::vector<int> A_rowInd;
//	std::vector<int> A_colInd;
//	std::vector<std::complex<double>> A_val;
//	for (size_t i = 0; i < A.rows(); i++) {
//		for (size_t j = 0; j < A.cols(); j++) {
//			A_val.push_back(A(i, j));
//			A_rowInd.push_back(i);
//			A_colInd.push_back(j);
//		}
//	}
//	int nnz = A_val.size();
//	int m = b.size();
//	//// Define device vectors
//	//thrust::device_vector<int> d_cooRowIndex, d_cooColIndex;
//	//thrust::device_vector<std::complex<double>> d_cooVal, d_b, d_x;
//	//d_cooRowIndex.resize(nnz);
//	//d_cooColIndex.resize(nnz);
//	//d_cooVal.resize(nnz);
//	//d_b.resize(m);
//	//d_x.resize(m);
//	//// Copy the COO formatted triplets to device
//	//thrust::copy(A_rowInd.begin(), A_rowInd.end(), d_cooRowIndex.begin());
//	//thrust::copy(A_colInd.begin(), A_colInd.end(), d_cooColIndex.begin());
//	//thrust::copy(A_val.begin(), A_val.end(), d_cooVal.begin());
//	//thrust::copy(&b[0],	&b[m- 1], d_b.begin());
//
//	//// Set raw pointers pointing to the triplets in the device
//	//int* r_cooRowIndex, * r_cooColIndex;
//	//std::complex<double>* r_cooVal, * r_b, * r_x;
//	//r_cooRowIndex = thrust::raw_pointer_cast(d_cooRowIndex.data());
//	//r_cooColIndex = thrust::raw_pointer_cast(d_cooColIndex.data());
//	//r_cooVal = thrust::raw_pointer_cast(d_cooVal.data());
//	//r_b = thrust::raw_pointer_cast(d_b.data());
//	//r_x = thrust::raw_pointer_cast(d_x.data());
//
//	//thrust::device_ptr<int> p_rowInd
//	//	= thrust::device_pointer_cast(r_cooRowIndex);
//	//thrust::device_ptr<int> p_colInd
//	//	= thrust::device_pointer_cast(r_cooColIndex);
//	//thrust::device_ptr<std::complex<double>> p_val = thrust::device_pointer_cast(r_cooVal);
//	//thrust::device_ptr<std::complex<double>> p_b = thrust::device_pointer_cast(r_b);
//	//thrust::device_ptr<std::complex<double>> p_x = thrust::device_pointer_cast(r_x);
//
//	//// use array1d_view to wrap the individual arrays
//	//typedef typename cusp::array1d_view<thrust::device_ptr<int>>
//	//	DeviceIndexArrayView;
//	//typedef typename cusp::array1d_view<thrust::device_ptr<std::complex<double>>>
//	//	DeviceValueArrayView;
//	//DeviceIndexArrayView row_indices(p_rowInd, p_rowInd + nnz);
//	//DeviceIndexArrayView column_indices(p_colInd, p_colInd + nnz);
//	//DeviceValueArrayView values(p_val, p_val + nnz);
//	//DeviceValueArrayView d_X(p_x, p_x + m);
//	//DeviceValueArrayView d_B(p_b, p_b + m);
//	//// combine the three array1d_views into a coo_matrix_view
//	//typedef cusp::coo_matrix_view<DeviceIndexArrayView,
//	//	DeviceIndexArrayView,
//	//	DeviceValueArrayView>
//	//	DeviceView;
//	//// construct a coo_matrix_view from the array1d_views
//	//DeviceView d_A(m, m, nnz, row_indices, column_indices, values);
//
//	//// set stopping criteria.
//	//int iteration_limit = 100;
//	//float relative_tolerance = 1e-15;
//	//bool verbose = false;  // Decide if the CUDA solver prints the iteration details or not
//	//cusp::monitor<cusp::complex<double>> monitor1(d_B, iteration_limit, relative_tolerance, verbose);
//
//	//// setup preconditioner
//	//cusp::precond::diagonal<float, cusp::device_memory> d_M(d_A);
//
//	//int restart = 50;
//	//// solve the linear system A * x = b with the BICGSTAB method
//	//cusp::krylov::gmres(d_A, d_X, d_B, restart, monitor1);
//
//	Eigen::VectorXcd x;
//	x.resize(m);
//	//cudaMemcpy(x.data(), r_x, sizeof(std::complex<double>) * m, cudaMemcpyDeviceToHost);
//	return x;
//}