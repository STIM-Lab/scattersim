#include <tira/optics/planewave.h>


/// <summary>
/// Write a list of plane waves representing single scattering events at a boundary. The resulting binary file can be used
/// to reconstruct the field along the surface.
/// </summary>
/// <typeparam name="T">Data type used to store the floating point values</typeparam>
/// <param name="filename">Name of the binary file to save</param>
/// <param name="plane_normal">Plane normal</param>
/// <param name="plane_position">A 3D point specifying a single point on the plane surface</param>
/// <param name="incident">STD vector of incident plane waves to save</param>
/// <param name="reflected">STD vector of reflected plane waves to save</param>
/// <param name="transmitted">STD vector of transmitted plane waves to save</param>
/// <returns></returns>
template <typename T>
int write_file(std::string filename,
	tira::vec3<T> plane_normal, tira::vec3<T> plane_position,
	std::vector< std::vector< tira::optics::planewave<T> > > incident,
	std::vector< std::vector< tira::optics::planewave<T> > > reflected,
	std::vector< std::vector< tira::optics::planewave<T> > > transmitted) {

	std::ofstream outfile(filename, std::ios::binary);							//create a binary output file for writing
	if (outfile) {																//if the file is successfully created
		outfile.write((char*)&plane_normal, sizeof(tira::vec3<T>));				//write the plane data first
		outfile.write((char*)&plane_position, sizeof(tira::vec3<T>));
		for (size_t b = 0; b < incident.size(); b++) {							//for each beam in the vector
			size_t N = incident[b].size();												//calculate the number of scattering events (i/r/t triples)
			outfile.write((char*)&N, sizeof(size_t));								//write the number of scattering events to the file

			for (size_t n = 0; n < N; n++) {												//for each scattering event
				outfile.write((char*)&incident[b][n], sizeof(tira::optics::planewave<T>));		//write the incident plane wave
				outfile.write((char*)&reflected[b][n], sizeof(tira::optics::planewave<T>));	//write the reflected plane wave
				outfile.write((char*)&transmitted[b][n], sizeof(tira::optics::planewave<T>));	//write the transmitted plane wave
			}
		}
		outfile.close();																//close the file
		return 0;																		//return
	}
	return -1;																			//if the file isn't created return -1
}

template <typename T>
int read_file(std::string filename,
	tira::vec3<T>& plane_normal, tira::vec3<T>& plane_position,
	std::vector< std::vector< tira::optics::planewave<T> > >& incident,
	std::vector< std::vector< tira::optics::planewave<T> > >& reflected,
	std::vector< std::vector< tira::optics::planewave<T> > >& transmitted) {

	std::ifstream infile(filename, std::ios::binary);							// create a binary input file for reading
	if (infile) {																// if the file exists
		infile.read((char*)&plane_normal, sizeof(tira::vec3<T>));
		infile.read((char*)&plane_position, sizeof(tira::vec3<T>));

		while (1) {
			size_t N = 0;														// initialize the number of scattering events
			infile.read((char*)&N, sizeof(size_t));								// read the number of scattering events
			std::vector< tira::optics::planewave<T> > tmp_i;					// temporary list of incident planewaves
			std::vector< tira::optics::planewave<T> > tmp_r;					// temporary list of reflected planewaves
			std::vector< tira::optics::planewave<T> > tmp_t;					// temporary list of transmitted planewaves
			if (N != 0) {
				tira::optics::planewave<T> tmp;							// temporary planewave
				for (size_t n = 0; n < N; n++) {
					infile.read((char*)&tmp, sizeof(tira::optics::planewave<T>));// read a new planewave
					tmp_i.push_back(tmp);
					infile.read((char*)&tmp, sizeof(tira::optics::planewave<T>));// read a new planewave
					tmp_r.push_back(tmp);
					infile.read((char*)&tmp, sizeof(tira::optics::planewave<T>));// read a new planewave
					tmp_t.push_back(tmp);
				}
				incident.push_back(tmp_i);
				reflected.push_back(tmp_r);
				transmitted.push_back(tmp_t);
			}
			else {
				break;
			}
		}
		infile.close();																
		return 0;																		
	}
	return -1;
}

template <typename T>
int write_file(std::string filename,
	tira::vec3<T> plane_normal, tira::vec3<T> plane_position,
	std::vector< tira::optics::planewave<T> > PWi,
	std::vector< tira::optics::planewave<T> > PWr,
	std::vector < tira::optics::planewave<T> > PWt) {

	std::vector< std::vector< tira::optics::planewave<T> > > Si(1, PWi);
	std::vector< std::vector< tira::optics::planewave<T> > > Sr(1, PWr);
	std::vector< std::vector< tira::optics::planewave<T> > > St(1, PWt);

	return write_file(filename, plane_normal, plane_position, Si, Sr, St);
}

template <typename T>
int write_file(std::string filename,
	tira::vec3<T> plane_normal, tira::vec3<T> plane_position,
	tira::optics::planewave<T> incident, tira::optics::planewave<T> reflected, tira::optics::planewave<T> transmitted) {

	std::vector< tira::optics::planewave<T> > PWi(1, incident);							//create a vector for a single plane wave
	std::vector< tira::optics::planewave<T> > PWr(1, reflected);
	std::vector< tira::optics::planewave<T> > PWt(1, transmitted);

	return write_file(filename, plane_normal, plane_position, PWi, PWr, PWt);			//call the code to save vectors of scattering events
}