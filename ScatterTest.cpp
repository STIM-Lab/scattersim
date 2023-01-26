#include <tira/optics/planewave.h>

#include <iostream>
int main(int argc, char** argv) {
	// Example 1
	// Notes: E0 is different from E1. But they should be the same, right?
	std::complex<double> Ex0(-8.39e-15, -8.4e-15);
	std::complex<double> Ey0(-1.51, 0.33);
	std::complex<double> Ez0(0.0, -0.0);

	std::complex<double> kx0(0.0);
	std::complex<double> ky0(0.0);
	std::complex<double> kz0(-6.28);

	tira::planewave p1(kx0, ky0, kz0, Ex0, Ey0, Ez0);
	glm::vec<3, std::complex<double>> E1 = p1.getE0();
	glm::vec<3, std::complex<double>> K1 = p1.getK();
	std::cout << "Example 1: " << std::endl;
	std::cout << "Input:" << std::endl;
	std::cout << "k = (" << kx0 << ", " << ky0 << ", " << kz0 << ")" << std::endl;
	std::cout<<"E0 = (" << Ex0 << ", " << Ey0 << ", " << Ez0 << ")" << std::endl;
	std::cout << "Correct Result:" << std::endl;
	std::cout << p1.str() << std::endl;
	std::cout << "Reinput the output to the system:" << std::endl;
	tira::planewave p10(K1[0], K1[1], K1[2], E1[0], E1[1], E1[2]);
	glm::vec<3, std::complex<double>> E10 = p10.getE0();
	std::cout << p10.str() << std::endl;

	std::cout << "Wrong Result:" << std::endl;
	tira::planewave p2(kx0, ky0, kz0, Ex0, Ey0);
	glm::vec<3, std::complex<double>> E2 = p2.getE0();
	std::cout << p2.str() << std::endl;


	// Example 2
	// Notes: E4 is obviously wrong
	std::complex<double> Ex1(7.36e-15, -1.10e-15);
	std::complex<double> Ey1(-0.43, 0.103);
	std::complex<double> Ez1(-7.36e-17, 1.10e-18);

	std::complex<double> kx1(-0.0628);
	std::complex<double> ky1(0.0);
	std::complex<double> kz1(-6.28);
	std::cout << "Example 2: " << std::endl;
	std::cout << "Input:" << std::endl;
	std::cout << "k = (" << kx1 << ", " << ky1 << ", " << kz1 << ")" << std::endl;
	std::cout << "E0 = (" << Ex1 << ", " << Ey1 << ", " << Ez1 << ")" << std::endl;
	std::cout << "Correct Result:" << std::endl;
	tira::planewave p3(kx1, ky1, kz1, Ex1, Ey1, Ez1);
	glm::vec<3, std::complex<double>> E3 = p3.getE0();
	glm::vec<3, std::complex<double>> K3 = p3.getK();
	std::cout << p3.str() << std::endl;
	std::cout << "Reinput the output to the system:" << std::endl;
	tira::planewave p8(K3[0], K3[1], K3[2], E3[0], E3[1], E3[2]);
	glm::vec<3, std::complex<double>> E8 = p8.getE0();
	std::cout << p8.str() << std::endl;

	std::cout << "Wrong Result:" << std::endl;
	tira::planewave p4(kx1, ky1, kz1, Ex1, Ey1);
	glm::vec<3, std::complex<double>> E4 = p4.getE0();
	std::cout << p4.str() << std::endl;


	// Example 3
	// Notes: E4 is obviously wrong
	std::complex<double> Ex2(0, 0);
	std::complex<double> Ey2(1, 0);
	std::complex<double> Ez2(0, 0);
						   
	std::complex<double> kx2(0);
	std::complex<double> ky2(0.0);
	std::complex<double> kz2(-6.28);

	std::cout << "Example 3: " << std::endl;
	std::cout << "Input:" << std::endl;
	std::cout << "k = (" << kx2 << ", " << ky2 << ", " << kz2 << ")" << std::endl;
	std::cout << "E0 = (" << Ex2 << ", " << Ey2 << ", " << Ez2 << ")" << std::endl;
	std::cout << "Correct Result:" << std::endl;
	tira::planewave p5(kx2, ky2, kz2, Ex2, Ey2, Ez2);
	glm::vec<3, std::complex<double>> E5 = p5.getE0();
	glm::vec<3, std::complex<double>> K5 = p5.getK();
	std::cout << p5.str() << std::endl;
	std::cout << "Reinput the output to the system:" << std::endl;
	tira::planewave p7(K5[0], K5[1], K5[2], E5[0], E5[1], E5[2]);
	glm::vec<3, std::complex<double>> E7 = p7.getE0();
	std::cout << p7.str() << std::endl;

	std::cout << "Wrong Result:" << std::endl;
	tira::planewave p6(kx2, ky2, kz2, Ex2, Ey2);
	glm::vec<3, std::complex<double>> E6 = p6.getE0();
	std::cout << p6.str() << std::endl;


	return 1;

}