/// This is an OpenGL "Hello World!" file that provides simple examples of inin_resolutiontegrating ImGui with GLFW
/// for basic OpenGL applications. The file also includes headers for the TIRA::GraphicsGL classes, which
/// provide an basic OpenGL front-end for creating materials and models for rendering.

#include "cpuEvaluator.h"
#include "tira/graphics_gl.h"
#include "tira/image/colormap.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"

#include <boost/program_options.hpp>

#include <cuda_runtime.h>
#include <extern/libnpy/npy.hpp>
#include "tira/cuda/error.h"
#include "gpuEvaluator.h"

#include <iostream>
#include <string>
#include <stdio.h>
#include <limits>
#include <complex>
#include <chrono>
#include <filesystem>
#include <regex>


GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
double window_width = 1600;
double window_height = 1200;
const char* glsl_version = "#version 130";              // specify the version of GLSL
ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);   // specify the OpenGL color used to clear the back buffer
float ui_scale = 1.5f;                                  // scale value for the UI and UI text
double xpos, ypos;                                      // cursor positions

float extent = 10;                                      // extent of the field being displayed (think of this as a zoom value)
float center[] = {0, 0, 0};                             // center of the display field
float float_high = 1000;                                // store the maximum float value
unsigned int res_step = 1;
float plane_position[] = {0.0f, 0.0f, 0.0f};

std::vector< glm::vec<3, std::complex<float>>* > E = { NULL, NULL, NULL };
//glm::vec<3, std::complex<float>>* E_xy;                  // stores the complex vector values for display
//glm::vec<3, std::complex<float>>* E_xz;
//glm::vec<3, std::complex<float>>* E_yz;

std::vector< std::complex<float>* > S = { NULL, NULL, NULL };   // store the complex scalar values for display

std::vector< tira::image<unsigned char> > I(3);
//tira::image<unsigned char> I_xy;                        // store the images for display
//tira::image<unsigned char> I_xz;
//tira::image<unsigned char> I_yz;

enum DisplayMode {X, Y, Z, Intensity};                  // display mode type
int display_mode = DisplayMode::X;                      // current display mode

bool show_real = true;
float real_min = -100;                                  // minimum real value in the field
float real_max = 100;                                   // maximum real value in the field
float real_low = real_min;
float real_high = real_max;
bool fix_low_high = false;
ImVec4 real_color = ImVec4(0.0f / 255.0f, 255.0f / 255.0f, 0.0f / 255.0f, 255.0f / 255.0f);

bool show_imag = true;
float imag_min = -100;                                  // minimum real value in the field
float imag_max = 100;                                   // maximum real value in the field
float imag_low = imag_min;
float imag_high = imag_max;
ImVec4 imag_color = ImVec4(0.0f / 255.0f, 0.0f / 255.0f, 255.0f / 255.0f, 255.0f / 255.0f);

bool use_colormap = false;                              // flag whether or not we use a pre-designed colormap
enum ColorMaps {Brewer, Magma, Grey};
const char* colormaps[] = { "Brewer", "Magma", "Grey" };
int colormap = ColorMaps::Brewer;
int colormap_component = 0;                             // 0 = real, 1 = imag

tira::glMaterial Material_xy;                           // OpenGL materials storing the texture and shader information for each slice
tira::glMaterial Material_yz;
tira::glMaterial Material_xz;
glm::mat4 projection;                                   // projection matrix for shader

tira::glGeometry SliceGeometry;

CoupledWaveStructure<double> cw;                        // coupled wave structure stores plane waves for the visualization
std::vector<std::string> in_filename;
std::string in_savename;
bool in_Visualization = true;                                // The filename for the output. Changeable by the cursor position.
int in_resolution;
//std::vector<int> in_slice;
int in_axis;
std::vector<float> in_center;
float in_slice;
// CUDA device information and management
int in_device;
float in_size;                                          // size of the sample being visualized (in arbitrary units specified during simulation)
bool in_intensity = false;                              // specify how multiple input files are combined (by default they are summed coherently)
size_t free_gpu_memory;
size_t total_gpu_memory;

bool verbose = false;
unsigned int in_isHete;

// time variables
std::chrono::time_point<std::chrono::steady_clock> start;
std::chrono::time_point<std::chrono::steady_clock> end;
double t_LoadData;
double t_AllocateCWStructure;
double t_UnpackCWStructure;
double t_SaveData;
double t_DeleteImageArrays;
double t_AllocateImageArrays;
double t_UpdateTextures;
double t_EvaluateColorSlices;
double t_EvaluateScalarSlices;
double t_EvaluateVectorSlices;
double t_MainFunction;
double t_InitCuda;
double t_UploadCudaData;

std::string VertexSource =                                  // Source code for the default vertex shader
"# version 330 core\n"

"layout(location = 0) in vec3 vertices;\n"
"layout(location = 2) in vec2 texcoords;\n"

"uniform mat4 MVP;\n"

"out vec4 vertex_color;\n"
"out vec2 vertex_texcoord;\n"

"void main() {\n"
"    gl_Position = MVP * vec4(vertices.x, vertices.y, vertices.z, 1.0f);\n"
"    vertex_texcoord = texcoords;\n"
"};\n";

std::string FragmentSource =
"# version 330 core\n"

"layout(location = 0) out vec4 color;\n"

"in vec4 vertex_color;\n"
"in vec2 vertex_texcoord;\n"
"uniform sampler2D texmap;\n"

"void main() {\n"
"    color = texture(texmap, vertex_texcoord);\n"
"};\n";



void DeleteImageArrays(){
    auto start = std::chrono::steady_clock::now();
    if(E[2] != NULL) delete E[2];
    if(E[1] != NULL) delete E[1];
    if(E[0] != NULL) delete E[0];

    if (S[2] != NULL) free(S[2]);
    if (S[1] != NULL) free(S[1]);
    if (S[0] != NULL) free(S[0]);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_DeleteImageArrays = duration.count();
}

void AllocateImageArrays(){
    DeleteImageArrays();

    auto start = std::chrono::steady_clock::now();
    size_t N = (size_t)pow(2, in_resolution);

    E[2] = (glm::vec<3, std::complex<float>>*)malloc(sizeof(glm::vec<3, std::complex<float>>) * N * N);
    E[1] = (glm::vec<3, std::complex<float>>*)malloc(sizeof(glm::vec<3, std::complex<float>>) * N * N);
    E[0] = (glm::vec<3, std::complex<float>>*)malloc(sizeof(glm::vec<3, std::complex<float>>) * N * N);

    S[2] = new std::complex<float>[N * N];
    S[1] = new std::complex<float>[N * N];
    S[0] = new std::complex<float>[N * N];

    I[2] = tira::image<unsigned char>(N, N, 3);
    I[1] = tira::image<unsigned char>(N, N, 3);
    I[0] = tira::image<unsigned char>(N, N, 3);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_AllocateImageArrays = duration.count();
}

void UpdateTextures() {
    auto start = std::chrono::steady_clock::now();
    Material_xy.SetTexture("texmap", I[2], GL_RGB, GL_NEAREST);
    Material_xz.SetTexture("texmap", I[1], GL_RGB, GL_NEAREST);
    Material_yz.SetTexture("texmap", I[0], GL_RGB, GL_NEAREST);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_UpdateTextures = duration.count();
}

float clamp(float v) {
    if (isnan(v)) return 0.0;
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}
glm::vec3 EvaluateColorValue(std::complex<float> v) {
    float r_interval = real_high - real_low;
    float i_interval = imag_high - imag_low;

    float r = v.real();
    float rn = (r - real_low) / r_interval;
    rn = clamp(rn);

    float i = v.imag();
    float in = (i - imag_low) / i_interval;
    in = clamp(in);

    glm::vec3 c(0, 0, 0);
    if (show_real) {
        c += glm::vec3(rn * real_color.x, rn * real_color.y, rn * real_color.z);
    }
    if (show_imag) {
        c += glm::vec3(in * imag_color.x, in * imag_color.y, in * imag_color.z);
    }
    return c;
}

void EvaluateColorSlices() {
    auto start = std::chrono::steady_clock::now();
    size_t N = pow(2, in_resolution);                                          // store the in_resolution of the field slices
    size_t N2 = N * N;
    float v;
    float n;
    float interval = real_high - real_low;

    // X-Y Color Evaluation
    glm::vec3 c;
    for (size_t yi = 0; yi < N; yi++) {
        for (size_t xi = 0; xi < N; xi++) {
            c = EvaluateColorValue(S[2][yi * N + xi]);
            I[2](xi, yi, 0) = c[0] * 255;
            I[2](xi, yi, 1) = c[1] * 255;
            I[2](xi, yi, 2) = c[2] * 255;
            
        }
    }

    // Y-Z Color Evaluation
    for (size_t zi = 0; zi < N; zi++) {
        for (size_t yi = 0; yi < N; yi++) {
            c = EvaluateColorValue(S[0][zi * N + yi]);
            I[0](yi, zi, 0) = c[0] * 255;
            I[0](yi, zi, 1) = c[1] * 255;
            I[0](yi, zi, 2) = c[2] * 255;
        }
    }

    // X-Z Color Evaluation
    for (size_t zi = 0; zi < N; zi++) {
        for (size_t xi = 0; xi < N; xi++) {
            c = EvaluateColorValue(S[1][zi * N + xi]);
            I[1](xi, zi, 0) = c[0] * 255;
            I[1](xi, zi, 1) = c[1] * 255;
            I[1](xi, zi, 2) = c[2] * 255;
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_EvaluateColorSlices = duration.count();
    UpdateTextures();
}

/// Calculate the minimum and maximum values for the scalar field and set the low and high values if specified
void CalculateMinMax() {
    size_t N = pow(2, in_resolution);                                          // store the resolution of the field slices
    size_t N2 = N * N;

    real_max = S[2][0].real();
    real_min = S[2][0].real();
    imag_max = S[2][0].imag();
    imag_min = S[2][0].imag();

    for (size_t i = 0; i < N2; i++) {
        if (S[2][i].real() > real_max) real_max = S[2][i].real();
        if (S[2][i].real() < real_min) real_min = S[2][i].real();
        if (S[2][i].imag() > imag_max) imag_max = S[2][i].imag();
        if (S[2][i].imag() < imag_min) imag_min = S[2][i].imag();
        if (S[0][i].real() > real_max) real_max = S[0][i].real();
        if (S[0][i].real() < real_min) real_min = S[0][i].real();
        if (S[0][i].imag() > imag_max) imag_max = S[0][i].imag();
        if (S[0][i].imag() < imag_min) imag_min = S[0][i].imag();
        if (S[1][i].real() > real_max) real_max = S[1][i].real();
        if (S[1][i].real() < real_min) real_min = S[1][i].real();
        if (S[1][i].imag() > imag_max) imag_max = S[1][i].imag();
        if (S[1][i].imag() < imag_min) imag_min = S[1][i].imag();
    }
    if(!fix_low_high){
        real_low = real_min;
        real_high = real_max;
        imag_low = imag_min;
        imag_high = imag_max;
    }
}

void EvaluateScalarCoordinate(int axis, int coord){

    size_t N = pow(2, in_resolution);                                          // store the resolution of the field slices
    size_t N2 = N * N;

    for (size_t i = 0; i < N2; i++) {
        S[axis][i] = E[axis][i][coord];
    }
}

void EvaluateScalarIntensity(int axis) {

    size_t N = pow(2, in_resolution);                                          // store the resolution of the field slices
    size_t N2 = N * N;

    for (size_t i = 0; i < N2; i++) {
        S[axis][i] = E[axis][i][0] * std::conj(E[axis][i][0]) +
            E[axis][i][1] * std::conj(E[axis][i][1]) +
            E[axis][i][2] * std::conj(E[axis][i][2]);
    }
}

void EvaluateScalarSlice(int axis) {
    if (display_mode == DisplayMode::Intensity) {
        EvaluateScalarIntensity(axis);
    }
    else {
        EvaluateScalarCoordinate(axis, display_mode);
    }
}

/// Selects scalar values for the field slices based on user input
void EvaluateScalarSlices() {
    auto start = std::chrono::steady_clock::now();

    size_t N = pow(2, in_resolution);                                          // store the resolution of the field slices
    size_t N2 = N * N;

    for (int a = 0; a < 3; a++)
        EvaluateScalarSlice(a);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_EvaluateScalarSlices = duration.count();
    CalculateMinMax();
    EvaluateColorSlices();
}

void EvaluateVectorSlices(int axis = -1, bool vector_only = false) {
    auto start = std::chrono::steady_clock::now();
    unsigned int N = pow(2, in_resolution);                                    // get the resolution of the field N
    float d = extent / (N - 1);                                             // calculate the step size in cartesian coordinates
    float x, y, z;
    float x_start = center[0] - extent / 2;
    float y_start = center[1] - extent / 2;
    float z_start = center[2] - extent / 2;
                                                                            // stores the plane index of the current pixel
    
    if (in_device >= 0)
        gpu_cw_evaluate((thrust::complex<float>*)E[2], (thrust::complex<float>*)E[1], (thrust::complex<float>*)E[0],
            x_start, y_start, z_start, plane_position[0], plane_position[1], plane_position[2], d, N, in_device, axis);
    else {
        cpu_cw_evaluate_xy(E[2], x_start, y_start, plane_position[2], d, N);
        cpu_cw_evaluate_xz(E[1], x_start, z_start, plane_position[1], d, N);
        cpu_cw_evaluate_yz(E[0], y_start, z_start, plane_position[0], d, N);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_EvaluateVectorSlices = duration.count();
    if(!vector_only)
        EvaluateScalarSlices();
}

void RenderGui(){
    ImGui::Begin("Display Controls");                                       // Create a window for all ImGui controls
    if(ImGui::DragFloat("Extent", &extent, 0.05, 0, float_high)){
        EvaluateVectorSlices();
    }
    if(ImGui::DragFloat3("Center", center, 0.05, -float_high, float_high)){
        EvaluateVectorSlices();
    }

    
    
    if (ImGui::GetIO().MouseClicked[1])
    {
        glfwGetCursorPos(window, &xpos, &ypos);
        ImGui::OpenPopup("save_slice");
    }

    if (ImGui::BeginPopup("save_slice"))
    {
        unsigned int N = pow(2, in_resolution);
        if (ImGui::Button("Save Slice")) {                                              // create a button that opens a file dialog
            ImGuiFileDialog::Instance()->OpenDialog("ChooseNpyFile", "Choose NPY File", ".npy,.npz", ".");              
        }
        if (ImGuiFileDialog::Instance()->Display("ChooseNpyFile")) {				    // if the user opened a file dialog
            if (ImGuiFileDialog::Instance()->IsOk()) {								    // and clicks okay, they've probably selected a file
                std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file
                std::string extension = filename.substr(filename.find_last_of(".") + 1);
                
                std::cout << "Cursor position: " << xpos << ", " << ypos << std::endl;
                std::cout << "File chosen: " << filename << std::endl;
                // RUIJIAO: determine which slice is clicked
                //          save the appropriate slice as an NPY file
                // Save the y-z slice
                if (xpos < window_width / 2.0 & ypos > window_height / 2.0) {
                    const std::vector<long unsigned> shape{ N, N};
                    const bool fortran_order{ false };
                    npy::SaveArrayAsNumpy(filename, fortran_order, shape.size(), shape.data(), S[0]);
                }
                // Save the x-y slice
                else if(xpos >= window_width / 2.0 & ypos < window_height / 2.0) {
                    const std::vector<long unsigned> shape{ N, N };
                    const bool fortran_order{ false };
                    npy::SaveArrayAsNumpy(filename, fortran_order, shape.size(), shape.data(), S[2]);

                }
                // Save the x-z slice
                else if (xpos >= window_width / 2.0 & ypos >= window_height / 2.0) {
                    const std::vector<long unsigned> shape{ N, N };
                    const bool fortran_order{ false };
                    npy::SaveArrayAsNumpy(filename, fortran_order, shape.size(), shape.data(), S[1]);

                }
                // Wrong click at the upper left region
                else {
                    std::cout << "Wrong click at the wrong region. " << std::endl;
                    exit(1);
                }
            }
            ImGuiFileDialog::Instance()->Close();									// close the file dialog box
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
    
    
    float min_plane[] = { center[0] - (extent * 0.5f), center[1] - (extent * 0.5f), center[2] - (extent * 0.5f) };
    float max_plane[] = { center[0] + (extent * 0.5f), center[1] + (extent * 0.5f), center[2] + (extent * 0.5f) };
    //ImGui::SliderScalarN("Plane Positions", ImGuiDataType_Float, plane_position, 3, min_plane, max_plane);
    ImGui::PushItemWidth(0.315 * ImGui::CalcItemWidth());
    if(ImGui::DragFloat("##PlaneX", &plane_position[0], extent * 0.001, center[0] - (extent * 0.5f), center[0] + (extent * 0.5f))){
        EvaluateVectorSlices();
    }
    ImGui::SameLine();
    if(ImGui::DragFloat("##PlaneY", &plane_position[1], extent * 0.001, center[1] - (extent * 0.5f), center[1] + (extent * 0.5f))){
        EvaluateVectorSlices();
    }
    ImGui::SameLine();
    if(ImGui::DragFloat("##PlaneZ", &plane_position[2], extent * 0.001, center[2] - (extent * 0.5f), center[2] + (extent * 0.5f))){
        EvaluateVectorSlices();
    }
    ImGui::SameLine();
    ImGui::Text("Planes");
    ImGui::PopItemWidth();

    ImGui::PushItemWidth(-ImGui::GetContentRegionAvail().x * 0.75f);
        if(ImGui::InputScalar("in_resolution = ", ImGuiDataType_U32, &in_resolution, &res_step, &res_step)){
            AllocateImageArrays();
            EvaluateVectorSlices();
        }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("%d x %d", (int)pow(2, in_resolution), (int)pow(2, in_resolution));

    if(ImGui::RadioButton("Ex(r)", &display_mode, DisplayMode::X)){
        EvaluateScalarSlices();
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("Ey(r)", &display_mode, DisplayMode::Y)){
        EvaluateScalarSlices();
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("Ez(r)", &display_mode, DisplayMode::Z) ){
        EvaluateScalarSlices();
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("I(r)", &display_mode, DisplayMode::Intensity)){
        EvaluateScalarSlices();
        show_imag = false;
    }

    if(ImGui::Checkbox("Use Colormap", &use_colormap)){
        EvaluateColorSlices();
    }
    ImGui::SameLine();
    if(!use_colormap) ImGui::BeginDisabled();
        ImGui::PushItemWidth(-ImGui::GetContentRegionAvail().x * 0.5f);
        const char* cmap_preview_value = colormaps[colormap];  // Pass in the preview value visible before opening the combo (it could be anything)
        if (ImGui::BeginCombo("##Select Colormap", cmap_preview_value))
        {
            for (int n = 0; n < IM_ARRAYSIZE(colormaps); n++)
            {
                const bool is_selected = (colormap == n);
                if (ImGui::Selectable(colormaps[n], is_selected)){
                    colormap = n;
                    EvaluateColorSlices();
                }

                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    if(!use_colormap) ImGui::EndDisabled();

    if(use_colormap){
        if(ImGui::RadioButton("Real", &colormap_component, 0)){
            EvaluateColorSlices();
        }
    }
    else{
        if(ImGui::Checkbox("Real", &show_real)){
            EvaluateColorSlices();
        }
    }
    ImGui::SameLine();
    if((use_colormap && colormap_component == 1) || (!use_colormap && !show_real))
        ImGui::BeginDisabled();
        if(ImGui::DragFloatRange2("##Range (Real)", &real_low, &real_high, (real_max - real_min)*0.01, real_min, real_max, "%f", "%f", ImGuiSliderFlags_AlwaysClamp)) {
            EvaluateColorSlices();
        }
    if((use_colormap && colormap_component == 1) || (!use_colormap && !show_real))
        ImGui::EndDisabled();
    ImGui::SameLine();
    if(use_colormap) ImGui::BeginDisabled();
        if(ImGui::ColorEdit3("##Real Color", (float*)&real_color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_Float)){
            EvaluateColorSlices();
        }
    if(use_colormap) ImGui::EndDisabled();
    
    if (display_mode == DisplayMode::Intensity) ImGui::BeginDisabled();
    if(use_colormap){
        if(ImGui::RadioButton("Imag", &colormap_component, 1)){
            EvaluateColorSlices();
        }
    }
    else{
        if(ImGui::Checkbox("Imag", &show_imag)){
            EvaluateColorSlices();
        }
    }
    ImGui::SameLine();
    if((use_colormap && colormap_component == 0) || (!use_colormap && !show_imag))
        ImGui::BeginDisabled();
        if(ImGui::DragFloatRange2("##Range (Imag)", &imag_low, &imag_high, (imag_max - imag_min) * 0.01, imag_min, imag_max, "%f", "%f", ImGuiSliderFlags_AlwaysClamp)){
            EvaluateColorSlices();
        }
    if((use_colormap && colormap_component == 0) || (!use_colormap && !show_imag))
        ImGui::EndDisabled();
    ImGui::SameLine();
    if(use_colormap) ImGui::BeginDisabled();
        if(ImGui::ColorEdit3("##Imag Color", (float*)&imag_color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_Float)){
            EvaluateColorSlices();
        }
    if(use_colormap) ImGui::EndDisabled();
    if (display_mode == DisplayMode::Intensity) ImGui::EndDisabled();
    ImGui::Checkbox("Fix Values", &fix_low_high);

    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    ImGui::Text("Performance");
    if(in_device >= 0){
        std::stringstream ss;
        ss << (float)free_gpu_memory / (1024.0f * 1000.0f) << "MB / " <<(float)total_gpu_memory / (1024.0f * 1000.0f) << "MB";
        float progress = (float)free_gpu_memory / (float)total_gpu_memory;
        ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), ss.str().c_str());
    }
    ImGui::Text("Layers: %d", (int)cw.Layers.size());
    ImGui::Text("     Layer 1: %d incident, %d reflected, %d transmitted", (int)cw.Pi.size(), (int)cw.Layers[0].Pr.size(), (int)cw.Layers[0].Pt.size());
    ImGui::Text("Load Data: %f s", t_LoadData);
    ImGui::Text("Evaluate Vector Fields: %f s", t_EvaluateVectorSlices);
    ImGui::Text("Evaluate Scalar Slices: %f s", t_EvaluateScalarSlices);
    ImGui::Text("Calculate Color Maps: %f s", t_EvaluateColorSlices);
    ImGui::Text("Evaluate Scalar Slices: %f s", t_EvaluateScalarSlices);
    ImGui::Text("Update Textures: %f s", t_UpdateTextures);
    ImGui::Text("Allocate Arrays: %f s", t_AllocateImageArrays);
    ImGui::Text("Delete Arrays: %f s", t_DeleteImageArrays);

    ImGui::End();                                                           // End rendering the "Hello, world!" window
}

/// <summary>
/// This function renders the user interface every frame
/// </summary>
void RenderUI() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Display a Demo window showing what ImGui is capable of
    // See https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html for code details
    //ImGui::ShowDemoWindow();

    RenderGui();


    ImGui::Render();                                                            // Render all windows
}

/// <summary>
/// Initialize the GUI
/// </summary>
/// <param name="window">Pointer to the GLFW window that will be used for rendering</param>
/// <param name="glsl_version">Version of GLSL that will be used</param>
void InitUI(GLFWwindow* window, const char* glsl_version) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(ui_scale);
    ImGui::GetIO().FontGlobalScale = ui_scale;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    //io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", ui_scale * 16.0f);

}

/// <summary>
/// Destroys the ImGui rendering interface (usually called when the program closes)
/// </summary>
void DestroyUI() {
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void TestCudaDevice(){
    start = std::chrono::steady_clock::now();
    // Initialize CUDA
    int nDevices;
    cudaError error = cudaGetDeviceCount(&nDevices);
    if (in_device > nDevices) {
        std::cout << "ERROR: CUDA device " << in_device << " unavailable (" << nDevices << " found)" << std::endl;
        std::cout << "*Use --cuda -1 for a CPU-only evaluation*" << std::endl;
        exit(1);
    }
    end = std::chrono::steady_clock::now();
    t_InitCuda = ((std::chrono::duration<double>)(end - start)).count();
   
    /*if (error != cudaSuccess || nDevices == 0) in_device = -1;                                                 // if there is an error getting device information, assume there are no devices

    if (in_device >= 0 && in_device < nDevices) {
        if (verbose) {
            std::cout << "Available CUDA Devices-----------------" << std::endl;
            for (int i = 0; i < nDevices; i++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                printf("Device Number: %d\n", i);
                printf("  Device name: %s\n", prop.name);
                printf("  Memory Clock Rate (KHz): %d\n",
                    prop.memoryClockRate);
                printf("  Memory Bus Width (bits): %d\n",
                    prop.memoryBusWidth);
                printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
                    2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
            }

            if (nDevices > in_device)
                std::cout << "Using Device " << in_device << " for data processing" << std::endl;
        }
    }*/
}
static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void OutputTimings() {
    int cw = 40;
    std::cout << std::setw(cw) << "Timing Breakdown" << std::endl;
    std::cout << std::setw(cw) << "------------------------------" << std::endl;
    std::cout << std::setw(cw) << "Total Time (Main Function): " << t_MainFunction << "s" << std::endl;
    std::cout << std::setw(cw) << "___________________________________" << std::endl;
    std::cout << std::setw(cw) << "     Loading Data: " << t_LoadData << "s" << std::endl;
    std::cout << std::setw(cw) << "     Allocating CW Struct: " << t_AllocateCWStructure << "s" << std::endl;
    std::cout << std::setw(cw) << "     Unpacking CW Struct: " << t_UnpackCWStructure << "s" << std::endl;
    std::cout << std::setw(cw) << "     Initializing CUDA: " << t_InitCuda << "s" << std::endl;
    std::cout << std::setw(cw) << "          Uploading CUDA Data: " << t_UploadCudaData << "s" << std::endl;
    std::cout << std::setw(cw) << "     Evaluating Vector Slices: " << t_EvaluateVectorSlices << "s" << std::endl;
    std::cout << std::setw(cw) << "     Evaluating Scalar Slices: " << t_EvaluateScalarSlices << "s" << std::endl;
    std::cout << std::setw(cw) << "     Evaluating Color Slices: " << t_EvaluateColorSlices << "s" << std::endl;
    std::cout << std::setw(cw) << "     Updating Textures: " << t_UpdateTextures << "s" << std::endl;
    std::cout << std::setw(cw) << "     Allocating Arrays: " << t_AllocateImageArrays << "s" << std::endl;
    std::cout << std::setw(cw) << "     Deleting Arrays: " << t_DeleteImageArrays << "s" << std::endl;
    std::cout << std::setw(cw) << "     Saving Data: " << t_SaveData << "s" << std::endl;
}

/// Loads and unpacks a CW file. If the GPU is being used, the unpacked CW data is also uploaded to the GPU
void UploadCW(std::string filename) {
    start = std::chrono::steady_clock::now();
    if (!cw.load(filename)) {                                     // load the coupled wave data
        std::cout << "ERROR: file " << in_filename[0] << " not found" << std::endl;
        exit(1);
    }
    end = std::chrono::steady_clock::now();
    t_LoadData = ((std::chrono::duration<double>)(end - start)).count();

    start = std::chrono::steady_clock::now();
    cw_allocate(&cw);
    end = std::chrono::steady_clock::now();
    t_AllocateCWStructure = ((std::chrono::duration<double>)(end - start)).count();

    start = std::chrono::steady_clock::now();
    cw_unpack(&cw);
    end = std::chrono::steady_clock::now();
    t_UnpackCWStructure = ((std::chrono::duration<double>)(end - start)).count();

    start = std::chrono::steady_clock::now();
    if (in_device >= 0)                                                          // if a CUDA device is specified
        gpu_upload_waves();                                                     // upload data to the GPU
    end = std::chrono::steady_clock::now();
    t_UploadCudaData = ((std::chrono::duration<double>)(end - start)).count();
}

void AccumulateIntensity(float* Isum, std::complex<float>* Inew, size_t N) {
    for (size_t n = 0; n < N; n++) {
        Isum[n] += Inew[n].real();
    }
}

void OutputField() {
    start = std::chrono::steady_clock::now();

    size_t i = 0;
    unsigned int N = pow(2, in_resolution);                             // N is the dimension of the image to be saved
    unsigned int N2 = N * N;                                            // N2 is the number of pixels in the image
    std::vector<float> I(N2, 0.0f);                                     // allocate an array to store the accumulated intensity

    if (in_intensity) {
        for (size_t i = 0; i < in_filename.size(); i++) {
            UploadCW(in_filename[i]);
            EvaluateVectorSlices(in_axis, true);                   // evaluate the vector slice (ONLY) for a single axis
            EvaluateScalarIntensity(in_axis);
            AccumulateIntensity(&I[0], &S[in_axis][0], N2);
        }
        const std::vector<long unsigned> shape{ N, N };
        const bool fortran_order{ false };
        npy::SaveArrayAsNumpy(in_savename, fortran_order, shape.size(), shape.data(), &I[0]);
    }

    /*
    if (in_axis == 1) {
        const std::vector<long unsigned> shape{ N, N, 3 };
        const bool fortran_order{ false };
        npy::SaveArrayAsNumpy(in_savename, fortran_order, shape.size(), shape.data(), (std::complex<float>*)E[1]);
    }
    // Save the x-y slice
    else if (in_axis == 2) {
        const std::vector<long unsigned> shape{ N, N, 3 };
        const bool fortran_order{ false };
        npy::SaveArrayAsNumpy(in_savename, fortran_order, shape.size(), shape.data(), (std::complex<float>*)E[2]);
    }
    // Save the yz slice
    else if (in_axis == 0) {
        const std::vector<long unsigned> shape{ N, N, 3 };
        const bool fortran_order{ false };
        npy::SaveArrayAsNumpy(in_savename, fortran_order, shape.size(), shape.data(), (std::complex<float>*)E[0]);
    }
    // Other cases
    else {
        std::cout << "ERROR: Invalid axis specified for output. " << std::endl;
        exit(1);
    }*/
    end = std::chrono::steady_clock::now();
    t_SaveData = ((std::chrono::duration<double>)(end - start)).count();
}

std::regex Wildcard2Regex(std::string arg) {
    for (auto i = arg.find('*'); i != std::string::npos; i = arg.find('*', i + 2)) {
        arg.insert(i, 1, '.');
    }

    return std::regex(arg);
}

/*std::filesystem::path FindFirstFile(std::filesystem::path directory,
                                    std::filesystem::path::const_iterator start, 
                                    std::filesystem::path::const_iterator finish, 
                                    std::string filename) {
    while (start != finish && start->string().find('*') == std::string::npos) {
        directory /= *start++;
    }
    std::filesystem::directory_iterator it(directory);
    std::filesystem::path result;

    if (it != std::filesystem::directory_iterator()) {
        if (start == finish) {
            for (auto i = filename.find('.'); i != std::string::npos; i = filename.find('.', i + 2)) {
                filename.insert(i, 1, '\\');
            }
            const auto re = Wildcard2Regex(filename);

            do {
                if (!std::filesystem::is_directory(it->status()) && std::regex_match(it->path().string(), re)) {
                    result = *it;
                    break;
                }
            } while (++it != std::filesystem::directory_iterator());
        }
        else {
            const auto re = Wildcard2Regex(start->string());

            do {
                if (it->is_directory() && std::regex_match(std::prev(it->path().end())->string(), re)) {
                    result = FindFirstFile(it->path(), next(start), finish, filename);

                    if (!result.empty()) {
                        break;
                    }
                }
            } while (++it != std::filesystem::directory_iterator());
        }
    }
    return result;
}*/

std::vector<std::string> FilesFromMask(std::string filemask) {
    std::filesystem::path fs_filemask(filemask);                                        // create a path object from the input mask
    std::string name_mask = fs_filemask.filename().string();                            // create a string representing the filename mask
    std::regex reg = Wildcard2Regex(name_mask);                                         // create a regex object based on the name mask

    std::filesystem::path working = std::filesystem::current_path();                    // store the current directory
    std::filesystem::path file_directory = fs_filemask.parent_path();                   // get the directory being searched
    if (file_directory.empty())                                                         // if the directory is empty (one wasn't provided)
        file_directory = working;                                                       // set it to the current working directory
    std::filesystem::path absolute_directory = std::filesystem::absolute(file_directory);      // get the directory relative to the current working directory

    std::vector<std::string> filenames;                                                                 // create a vector to store all of the matching file names



    for (const auto& entry : std::filesystem::directory_iterator(absolute_directory)) {                 // for each file in the relative directory
        std::cout << entry.path().string() << std::endl;
        std::string candidate = entry.path().filename().string();                                       // save the candidate filename as a string
        if (std::regex_match(candidate, reg))                                                           // see if the candidate filename matches the regex object
            filenames.push_back(std::filesystem::absolute(entry.path()).string());                      // if it does, push it to the filenames vector
    }
    return filenames;
}


int main(int argc, char** argv) {
    auto start_m = std::chrono::steady_clock::now();
    boost::program_options::options_description desc("Allowed options");
	desc.add_options()
        //("input", boost::program_options::value<std::vector< std::string> >(&in_filename)->multitoken()->default_value(std::vector<std::string>{"psf.cw"}, "psf.cw"), "input filename(s)")
        ("input", boost::program_options::value<std::vector< std::string> >(&in_filename), "input filename(s)")
        ("help", "produce help message")
        ("cuda", boost::program_options::value<int>(&in_device)->default_value(0), "cuda device number (-1 is CPU-only)")
        ("nogui", "save an output file without loading the GUI")
		("verbose", "produce verbose output")
        ("sample", "load a 3D sample stored as a grid (*.npy)")
        ("size", boost::program_options::value<float>(&in_size)->default_value(10), "size of the sample being visualized (initial range in arbitrary units)")
        ("resolution", boost::program_options::value<int>(&in_resolution)->default_value(8), "resolution of the sample field (use powers of two, ex. 2^n)")
        ("output", boost::program_options::value<std::string>(&in_savename), "output file (optional)")
        ("axis", boost::program_options::value<int>(&in_axis)->default_value(1), "axis to cut (0 = X, 1 = Y, 2 = Z")
        ("center", boost::program_options::value<std::vector<float> >(&in_center)->multitoken()->default_value(std::vector<float>{0, 0, 0}, "{0, 0, 0}"), "center position of the sampled volume")
        ("slice", boost::program_options::value<float>(&in_slice)->default_value(0), "coordinate along the specified axis RELATIVE to the 'center' position")
        ("intensity", "combine multiple input files inhoherently by summing their intensities")
		;
	boost::program_options::variables_map vm;

    boost::program_options::positional_options_description p;
    p.add("input", -1);
    //boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

    boost::program_options::parsed_options parsed = boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run();
   // boost::program_options::parsed_options parsed = boost::program_options::parse_command_line(argc, argv, desc, boost::program_options::command_line_style::unix_style ^ boost::program_options::command_line_style::allow_short);

    boost::program_options::store(parsed, vm);
	//boost::program_options::store()
    boost::program_options::notify(vm); 

    extent = in_size;                           // initialize the extent of the visualization to the size of the sample
  

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}
    if (vm.count("nogui"))
        in_Visualization = false;

    if (vm.count("intensity"))
        in_intensity = true;

    // set the initial plane position based on the command line arguments
    if (in_axis == 0)
        plane_position[0] = in_slice;
    else if (in_axis == 1)
        plane_position[1] = in_slice;
    else if (in_axis == 2)
        plane_position[2] = in_slice;

    center[0] = in_center[0];
    center[1] = in_center[1];
    center[2] = in_center[2];


    if(vm.count("verbose")){
        verbose = true;
    }
    
    if(!vm.count("input")){                                             // load the input file and check for errors
        std::cout << "ERROR: no input file specified" << std::endl;
        exit(1);
    }

    //std::vector< std::filesystem::path > filepaths;
    if (in_filename.size() == 1) {
        in_filename = FilesFromMask(in_filename[0]);
    }

    
    if (in_device >= 0)
        TestCudaDevice();                                               // test the CUDA device (if its being used)


    AllocateImageArrays();                                              // allocate space to store the evaluated fields

    if (vm.count("output"))                                             // if the user specified an output file
        OutputField();                                                  // output the field as a numpy file
    else                                                                // otherwise the user is just visualizing the field
        UploadCW(in_filename[0]);                                       // load the first CW file for visualization
    
                          
    // Output Timings for calculating the output file
    if (verbose && vm.count("output")) {

        auto end_m = std::chrono::steady_clock::now();
        t_MainFunction = ((std::chrono::duration<double>)(end_m - start_m)).count();
        OutputTimings();
    }

    if (in_Visualization == false) return 0;                            // if the GUI isn't active, we're done

    

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;
    // GL 3.0 + GLSL 130
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create window with graphics context
    std::string window_title = "ScatterView - " + in_filename[0];
    window = glfwCreateWindow(window_width, window_height, window_title.c_str(), NULL, NULL);
    if (window == NULL)
        return 1;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        return 0;
    }
    

    InitUI(window, glsl_version);

    
    Material_xy.CreateShader(VertexSource, FragmentSource);         // create a material based on the vertex and fragment shaders
    Material_xz.CreateShader(VertexSource, FragmentSource);
    Material_yz.CreateShader(VertexSource, FragmentSource);  

    SliceGeometry = tira::glGeometry::GenerateRectangle<float>();
    
    EvaluateVectorSlices();

    

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        RenderUI();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        
        float aspect = (float)display_w / (float)display_h;
        if (aspect > 1)
            projection = glm::ortho(-0.5 * aspect, 0.5 * aspect, 0.5, -0.5);
        else
            projection = glm::ortho(-0.5, 0.5, 0.5 * (1.0 / aspect), -0.5 * (1.0 / aspect));

        

        glClear(GL_COLOR_BUFFER_BIT);                               // clear the Viewport using the clear color

        /****************************************************/
        /*      Draw Stuff To The Viewport                  */
        /****************************************************/
        glViewport(0, 0, display_w / 2, display_h / 2);                     // specifies the area of the window where OpenGL can render
        Material_yz.Begin();
        Material_yz.SetUniformMat4f("MVP", projection);
        SliceGeometry.Draw();
        Material_yz.End();

        glViewport(display_w / 2, display_h / 2, display_w / 2, display_h / 2);                     // specifies the area of the window where OpenGL can render
        Material_xy.Begin();
        Material_xy.SetUniformMat4f("MVP", projection);
        SliceGeometry.Draw();
        Material_xy.End();

        glViewport(display_w / 2, 0, display_w / 2, display_h / 2);                     // specifies the area of the window where OpenGL can render
        Material_xz.Begin();
        Material_xz.SetUniformMat4f("MVP", projection);
        SliceGeometry.Draw();
        Material_xz.End();



        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer

        glfwSwapBuffers(window);                                    // swap the double buffer
    }

    DeleteImageArrays();
    DestroyUI();                                                    // Clear the ImGui user interface

    glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
    glfwTerminate();                                                // Terminate GLFW

    return 0;
}