#include <iostream>
#include <tira/optics/planewave.h>
#include "CoupledWaveStructure.h"
//#include <tira/optics/beam.h>
//#include <tira/geometry/plane.h>
//#include <numbers>
#include <complex>
#include <math.h>
//#include <tira/geometry/vec3.h>
//#include "ScatterIO.h"
#include <fstream>
#include <boost/program_options.hpp>
#include <random>
#include <iomanip>
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

std::vector<double> in_dir;
double in_lambda;
double in_kappa;
std::vector<double> in_n;
std::vector<double> in_ex;
std::vector<double> in_ey;
double in_z;
std::vector<double> in_normal;
std::string in_outfile;
double in_alpha;
double in_beta;
std::vector<unsigned int> in_samples;
std::string in_mode;

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

std::string vec2str(glm::vec<3, double> v, int spacing = 20) {
	std::stringstream ss;
	ss << std::setw(spacing) << std::left << v[0] << std::setw(spacing) << std::left << v[1] << std::setw(spacing) << std::left << v[2];
	return ss.str();
}

/// Removes waves with a k-vector pointed along the negative z axis
std::vector< tira::planewave<double> > RemoveInvalidWaves(std::vector<tira::planewave<double>> W){
	std::vector<tira::planewave<double>> new_W;
	for(size_t i = 0; i < W.size(); i++){
		if(W[i].getKreal()[2] >0)
			new_W.push_back(W[i]);
	}

	return new_W;
}


int main(int argc, char** argv) {

	// Declare the supported options.
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("lambda,l", boost::program_options::value<double>(&in_lambda)->default_value(1.0), "incident field vacuum wavelength")
		("direction,d", boost::program_options::value<std::vector<double> >(&in_dir)->multitoken()->default_value(std::vector<double>{1, 0, 1}, "0, 0, 1"), "incoming field direction")
		("ex", boost::program_options::value<std::vector<double> >(&in_ex)->multitoken()->default_value(std::vector<double>{0, 0}, "0 0"), "incoming field direction")
		("ey", boost::program_options::value<std::vector<double> >(&in_ey)->multitoken()->default_value(std::vector<double>{1, 0}, "1 0"), "incoming field direction")
		("n", boost::program_options::value<std::vector<double>>(&in_n)->multitoken()->default_value(std::vector<double>{1.0, 1.4}, "1.0 1.4"), "layer refractive indices")
		("kappa", boost::program_options::value<double>(&in_kappa)->default_value(0.05), "transmitted material absorption coefficient")
		("z,z", boost::program_options::value<double>(&in_z)->default_value(0.0), "position of the plane along the z axis")
		("output,o", boost::program_options::value<std::string>(&in_outfile)->default_value("a.cw"), "output filename for the coupled wave structure")
		("alpha,a", boost::program_options::value<double>(&in_alpha)->default_value(0.5), "angle used to focus the incident field")
		("beta,b", boost::program_options::value<double>(&in_beta)->default_value(0.0), "internal obscuration angle (for simulating reflective optics)")
		("samples,s", boost::program_options::value<std::vector<unsigned int> >(&in_samples)->multitoken()->default_value(std::vector<unsigned int>{64, 64}, "375"), "number of samples (can be specified in 2 dimensions)")
		("mode,m", boost::program_options::value<std::string>(&in_mode)->default_value("polar"), "sampling mode (polar, montecarlo)")
		("log", "produce a log file")
		;
	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm); 

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	std::ofstream logfile;										// if a log is requested, begin output
	if(vm.count("log")){
		std::stringstream ss;
		ss<<std::time(0)<<"_scatterplane.log";
		logfile.open(ss.str());
	}

	glm::tvec3<double> dir = glm::normalize(glm::tvec3<double>(in_dir[0], in_dir[1], in_dir[2]));				// set the direction of the incoming source field
	double k = 2 * M_PI / (in_lambda * in_n[0]);
	
	
	glm::tvec3<double> n = glm::normalize(glm::tvec3<double>(0, 0, -1));
	std::complex<double> ni(in_n[0], 0.0);
	std::complex<double> nt(in_n[1], in_kappa);

	std::string filename = in_outfile;

	glm::tvec3<double> p(0, 0, in_z);
	tira::planewave<double> r;
	tira::planewave<double> t;	
	std::complex<double> nr = nt/ni;
	tira::planewave<double> i(k * dir[0], k * dir[1], k * dir[2], std::complex<double>(in_ex[0], in_ex[1]), std::complex<double>(in_ey[0], in_ey[1]));
	glm::vec<3, std::complex<double> > E0 = i.getE0();
	unsigned int N[2];										// calculate the number of samples

	if (in_samples.size() == 1) {
		if (in_mode == "montecarlo") {
			N[0] = in_samples[0];
			N[1] = 1;
		}
		else {
			N[0] = N[1] = std::sqrt(in_samples[0]);
		}
	}
	else {
		N[0] = in_samples[0];
		N[1] = in_samples[1];
	}
	if (in_alpha == 0) {
		N[0] = 1;
		N[1] = 1;
	}


	i.scatter(n, p, nr, r, t);														// create one representative plane wave (also used if NA = 0 or N = 1)

	int spacing1 = 30;
	int spacing2 = 30;

	// incident field parameters
	std::cout << std::setw(spacing1) << std::left << "vacuum wavelength: " << in_lambda << std::endl;

	// optics
	if (in_alpha != 0.0) {
		std::cout << std::setw(spacing1) << std::left << "focusing angle: " << in_alpha << " (" << std::sin(in_alpha) * ni.real() << " NA)" << std::endl;
		if(in_beta > 0.0)
			std::cout << std::setw(spacing1) << std::left << "obscuration angle: " << in_beta << " (" << std::sin(in_beta) * ni.real() << " NA)" << std::endl;
	}
	std::cout << std::setw(spacing1) << std::left << "samples: " << N[0] << " x " << N[1] << " = " << N[0] * N[1] << std::endl;
	std::cout << std::setw(spacing1) << std::left << "sampling mode: " << in_mode << std::endl;
	std::cout << std::endl;

	std::cout << std::setw(spacing1) << std::left << "↓↓↓↓↓   k:" << vec2str(i.getK(), spacing2) << std::endl;
	glm::vec<3, std::complex<double>> i_E = i.getE0();
	std::cout << std::setw(spacing1) << std::left << "↓↓↓↓↓   E(0):" << vec2str(i_E, spacing2) << std::endl << std::endl;

	glm::vec<3, std::complex<double>> r_k = r.getK();
	std::cout << std::setw(spacing1) << std::left << "↑↑↑↑↑   k:" << vec2str(r_k, spacing2) << std::endl;
	glm::vec<3, std::complex<double>> r_E = r.getE0();
	std::cout << std::setw(spacing1) << std::left << "↑↑↑↑↑   E(0):" << vec2str(r_E, spacing2) << std::endl;

	std::cout << std::endl;
	std::cout << "----------------------------n = " << ni.real() <<" + "<< 0.0 <<"i----------------------------"<< std::endl;
	std::cout << "                            z = " << in_z << std::endl;
	std::cout << "----------------------------n = " << nt.real() << " + "<<nt.imag()<<"i" << std::endl;
	std::cout << std::endl;

	glm::vec<3, std::complex<double>> t_k = t.getK();
	std::cout << std::setw(spacing1) << std::left << "↓↓↓↓↓ k:" << vec2str(t_k, spacing2) << std::endl;
	glm::vec<3, std::complex<double>> t_E = t.getE0();
	std::cout << std::setw(spacing1) << std::left << "↓↓↓↓↓ E(0):" << vec2str(t_E, spacing2) << std::endl;
	std::cout << std::endl << std::endl;

	std::cout << std::setw(spacing1) << std::left << "output file: " << filename << std::endl;

	// allocate a coupled wave structure to store simulation results
	CoupledWaveStructure<double> cw;
	//cw.Pi.resize(N[0] * N[1]);
	cw.Layers.resize(1);
	cw.Layers[0].z = p[2];
	//cw.Layers[0].Pr.resize(N[0] * N[1]);
	//cw.Layers[0].Pt.resize(N[0] * N[1]);

	if (in_alpha == 0 || N[0] * N[1] == 1) {																			// if there is only one plane, save the previous simulation
		cw.Pi.push_back(i);
		cw.Layers[0].Pr.push_back(r);
		cw.Layers[0].Pt.push_back(t);
		if (logfile) {
			logfile << "i (" << 0 << ") ------------" << std::endl << i.str() << std::endl;
			logfile << "r (" << 0 << ") ------------" << std::endl << r.str() << std::endl;
			logfile << "t (" << 0 << ") ------------" << std::endl << t.str() << std::endl;
			logfile << std::endl;
		}
	}
	else {
		std::vector< tira::planewave<double> > I;
		if(in_mode == "montecarlo")
			I = tira::planewave<double>::SolidAngleMC(in_alpha, k * dir[0], k * dir[1], k * dir[2], std::complex<double>(in_ex[0], in_ex[1]), std::complex<double>(in_ey[0], in_ey[1]), N[0] * N[1], in_beta, n);
		else if(in_mode == "polar")
			I = tira::planewave<double>::SolidAnglePolar(in_alpha, k * dir[0], k * dir[1], k * dir[2], std::complex<double>(in_ex[0], in_ex[1]), std::complex<double>(in_ey[0], in_ey[1]), N[0], N[1], in_beta, n);
		
		// Some waves may be directed upwards from the surface, due to a combination of a large NA and low angle of incidence
		// This function removes waves with negative z k-vector components
		//if(logfile){
		//	logfile<<"Number of waves produced: "<<I.size()<<std::endl;
		//}
		//I = RemoveInvalidWaves(I);	
		//if(logfile){
		//	logfile<<"Number of waves simulated: "<<I.size()<<std::endl<<std::endl;
		//}	

		for (size_t idx = 0; idx < I.size(); idx++) {
			i = I[idx];
			i.scatter(n, p, nr, r, t);
			cw.Pi.push_back(i);
			cw.Layers[0].Pr.push_back(r);
			cw.Layers[0].Pt.push_back(t);
			if(logfile){
				logfile<<"i ("<<idx<<") ------------"<<std::endl<<i.str()<<std::endl;
				logfile<<"r ("<<idx<<") ------------"<<std::endl<<r.str()<<std::endl;
				logfile<<"t ("<<idx<<") ------------"<<std::endl<<t.str()<<std::endl;
				logfile<<std::endl;
			}
		}
	}

	if(logfile) logfile.close();

	if (filename != "") {		
		cw.save(filename);
	}
}