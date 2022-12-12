#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 3.19)

#Name your project here
project(scatter LANGUAGES CUDA CXX)
#project(scatter LANGUAGES CXX)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)

#set up CUDA
enable_language(CUDA)

#find the TIRA library
find_package(Boost REQUIRED COMPONENTS program_options)
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



#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif ( MSVC )

#set the include directories
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${TIRA_INCLUDE_DIRS}
			${CUDA_INCLUDE_DIRS}
			${Boost_INCLUDE_DIRS}
)

				
#create an executable
add_executable(scatterplane
					CoupledWaveStructure.h
					# HeteSaveStructure.h
					ScatterPlane.cpp
					cnpy/cnpy.cpp)

add_executable(scatterlayer
					CoupledWaveStructure.h
					# HeteSaveStructure.h
					ScatterLayer.cpp)

add_executable(scattervolume
					CoupledWaveStructure.h
					# HeteSaveStructure.h
					FourierWave.h
					# CoupledWaveVolumeStructure.h
					ScatterVolume.cpp)

add_executable(scatterview
					CoupledWaveStructure.h
					cpuEvaluator.cpp
					cpuEvaluator.h
					gpuEvaluator.cu
					gpuEvaluator.h
					# HeteSaveStructure.h
					# HetecpuEvaluator.cpp
					# HetecpuEvaluator.h
					# HetegpuEvaluator.cu
					# HetegpuEvaluator.h
					ScatterView.cpp)

# Final project to finish
add_executable(scatterviewHete
					CoupledWaveStructureHete.h
					cpuEvaluatorHete.cpp
					cpuEvaluatorHete.h
					gpuEvaluatorHete.cu
					gpuEvaluatorHete.h
					ScatterViewHete.cpp)

target_link_libraries(scatterplane
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

target_link_libraries(scattervolume
			ZLIB::ZLIB
			glm::glm
			Boost::program_options
			Eigen3::Eigen
			FFTW3::fftw3
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

target_link_libraries(scatterviewHete
			ZLIB::ZLIB
			glfw
			glm::glm
			GLEW::GLEW
			${OPENGL_LIBRARIES}
			imgui::imgui
			Boost::program_options
)