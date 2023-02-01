#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 3.16)

#Name your project here
project(scatter LANGUAGES CUDA CXX)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)

# fix warnings with BOOST and CMake
set(Boost_NO_WARN_NEW_VERSIONS 1)
find_package(Boost REQUIRED COMPONENTS program_options)


#find the TIRA library
set(CMAKE_CUDA_ARCHITECTURES 75)
find_package(CUDA REQUIRED)


find_package(glm CONFIG REQUIRED)
find_package(TIRA REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED core glfw-binding opengl3-binding)
find_package(GLEW REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)
find_package(MKL CONFIG REQUIRED)


#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif ( MSVC )

configure_file(data/sphere.npy 
				sphere.npy COPYONLY)
configure_file(data/RecSquare.npy 
				RecSquare.npy COPYONLY)
configure_file(data/target.npy 
				target.npy COPYONLY)
configure_file(data/triangle.npy 
				triangle.npy COPYONLY)

#set the include directories
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${TIRA_INCLUDE_DIRS}
			${CUDA_INCLUDE_DIRS}
			${Boost_INCLUDE_DIRS}
			${MKL_H}
)
link_directories(${CUDA_BIN_DIRS})
				
#create an executable
add_executable(scatterplane
					CoupledWaveStructure.h
					# HeteSaveStructure.h
					ScatterPlane.cpp
					cnpy/cnpy.cpp)

add_executable(scattertest
					CoupledWaveStructure.h
					ScatterTest.cpp
					cnpy/cnpy.cpp)

add_executable(scatterlayer
					CoupledWaveStructure.h
					# HeteSaveStructure.h
					ScatterLayer.cpp)

add_executable(scatterthinvolume
					CoupledWaveStructure.h
					FourierWave.h
					ScatterThinVolume.cpp
					third_Lapack.h)
add_executable(scattervolume
					CoupledWaveStructure.h
					FourierWave.h
					ScatterVolume.cpp
					third_Lapack.h)

add_executable(scatterview
					CoupledWaveStructure.h
					cpuEvaluator.cpp
					cpuEvaluator.h
					gpuEvaluator.cu
					gpuEvaluator.h
					ScatterView.cpp)

target_link_libraries(scatterplane
			ZLIB::ZLIB
			glm::glm
			Boost::program_options
)

target_link_libraries(scattertest
			ZLIB::ZLIB
			glm::glm
			Boost::program_options
)

target_link_libraries(scatterlayer
			ZLIB::ZLIB
			glm::glm
			Boost::program_options
			Eigen3::Eigen
)

target_link_libraries(scatterthinvolume
			ZLIB::ZLIB
			glm::glm
			Boost::program_options
			Eigen3::Eigen
			FFTW3::fftw3
			${CUDA_LIBRARIES}
			MKL::MKL
)

target_link_libraries(scattervolume
			ZLIB::ZLIB
			glm::glm
			Boost::program_options
			Eigen3::Eigen
			FFTW3::fftw3
			${CUDA_LIBRARIES}
			MKL::MKL
)

target_link_libraries(scatterview
			ZLIB::ZLIB
			glfw
			glm::glm
			GLEW::GLEW
			${OPENGL_LIBRARIES}
			imgui::imgui
			Boost::program_options
)