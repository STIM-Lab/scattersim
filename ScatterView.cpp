/// This is an OpenGL "Hello World!" file that provides simple examples of integrating ImGui with GLFW
/// for basic OpenGL applications. The file also includes headers for the TIRA::GraphicsGL classes, which
/// provide an basic OpenGL front-end for creating materials and models for rendering.

#include "cpuEvaluator.h"
#include "tira/graphics_gl.h"
#include "CoupledWaveStructure.h"
#include "tira/image/colormap.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <boost/program_options.hpp>

#include <cuda_runtime.h>
#include "tira/cuda/error.h"
#include "gpuEvaluator.h"

#include <iostream>
#include <string>
#include <stdio.h>
#include <limits>
#include <complex>
#include <chrono>


GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
const char* glsl_version = "#version 130";              // specify the version of GLSL
ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);   // specify the OpenGL color used to clear the back buffer
float ui_scale = 1.5f;                                  // scale value for the UI and UI text


float extent = 10;                                      // extent of the field being displayed (think of this as a zoom value)
float center[] = {0, 0, 0};                             // center of the display field
float float_high = 1000;                                // store the maximum float value
unsigned int resolution = 8;                            // resolution of the sample field (use powers of two, ex. 2^n)
unsigned int res_step = 1;
float plane_position[] = {0.0f, 0.0f, 0.0f};

glm::vec<3, std::complex<float>>* E_xy;                  // stores the complex vector values for display
glm::vec<3, std::complex<float>>* E_xz;
glm::vec<3, std::complex<float>>* E_yz;

std::complex<float>* S_xy = NULL;                       // store the complex scalar values for display
std::complex<float>* S_xz = NULL;
std::complex<float>* S_yz = NULL;

tira::image<unsigned char> I_xy;                        // store the images for display
tira::image<unsigned char> I_xz;
tira::image<unsigned char> I_yz;

enum DisplayMode {X, Y, Z, Intensity};                  // display mode type
int display_mode = DisplayMode::X;                      // current display mode

bool show_real = true;
float real_min = -100;                                  // minimum real value in the field
float real_max = 100;                                   // maximum real value in the field
float real_low = real_min;
float real_high = real_max;
bool fix_low_high = false;
ImVec4 real_color = ImVec4(255.0f / 255.0f, 0.0f / 255.0f, 0.0f / 255.0f, 255.0f / 255.0f);

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
std::string in_filename;

// CUDA device information and management
int in_device;
size_t free_gpu_memory;
size_t total_gpu_memory;

bool verbose = false;

// time variables
double t_LoadData;
double t_DeleteImageArrays;
double t_AllocateImageArrays;
double t_UpdateTextures;
double t_EvaluateColorSlices;
double t_EvaluateScalarSlices;
double t_EvaluateVectorSlices;


void DeleteImageArrays(){
    auto start = std::chrono::steady_clock::now();
    if(E_xy != NULL) delete E_xy;
    if(E_xz != NULL) delete E_xz;
    if(E_yz != NULL) delete E_yz;

    if(S_xy != NULL) free(S_xy);
    if(S_xz != NULL) free(S_xz);
    if(S_yz != NULL) free(S_yz);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_DeleteImageArrays = duration.count();
}

void AllocateImageArrays(){
    DeleteImageArrays();

    auto start = std::chrono::steady_clock::now();
    size_t N = (size_t)pow(2, resolution);

    E_xy = (glm::vec<3, std::complex<float>>*)malloc(sizeof(glm::vec<3, std::complex<float>>) * N * N);
    E_xz = (glm::vec<3, std::complex<float>>*)malloc(sizeof(glm::vec<3, std::complex<float>>) * N * N);
    E_yz = (glm::vec<3, std::complex<float>>*)malloc(sizeof(glm::vec<3, std::complex<float>>) * N * N);

    S_xy = new std::complex<float>[N * N];
    S_xz = new std::complex<float>[N * N];
    S_yz = new std::complex<float>[N * N];

    I_xy = tira::image<unsigned char>(N, N, 3);
    I_xz = tira::image<unsigned char>(N, N, 3);
    I_yz = tira::image<unsigned char>(N, N, 3);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_AllocateImageArrays = duration.count();
}

void UpdateTextures() {
    auto start = std::chrono::steady_clock::now();
    Material_xy.SetTexture("texmap", I_xy, GL_RGB, GL_NEAREST);
    Material_xz.SetTexture("texmap", I_xz, GL_RGB, GL_NEAREST);
    Material_yz.SetTexture("texmap", I_yz, GL_RGB, GL_NEAREST);
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
    size_t N = pow(2, resolution);                                          // store the resolution of the field slices
    size_t N2 = N * N;
    float v;
    float n;
    float interval = real_high - real_low;

    // X-Y Color Evaluation
    glm::vec3 c;
    for (size_t yi = 0; yi < N; yi++) {
        for (size_t xi = 0; xi < N; xi++) {
            c = EvaluateColorValue(S_xy[yi * N + xi]);
            I_xy(xi, yi, 0) = c[0] * 255;
            I_xy(xi, yi, 1) = c[1] * 255;
            I_xy(xi, yi, 2) = c[2] * 255;
            
        }
    }

    // Y-Z Color Evaluation
    for (size_t zi = 0; zi < N; zi++) {
        for (size_t yi = 0; yi < N; yi++) {
            c = EvaluateColorValue(S_yz[zi * N + yi]);
            I_yz(yi, zi, 0) = c[0] * 255;
            I_yz(yi, zi, 1) = c[1] * 255;
            I_yz(yi, zi, 2) = c[2] * 255;
        }
    }

    // X-Z Color Evaluation
    for (size_t zi = 0; zi < N; zi++) {
        for (size_t xi = 0; xi < N; xi++) {
            c = EvaluateColorValue(S_xz[zi * N + xi]);
            I_xz(xi, zi, 0) = c[0] * 255;
            I_xz(xi, zi, 1) = c[1] * 255;
            I_xz(xi, zi, 2) = c[2] * 255;
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_EvaluateColorSlices = duration.count();
    UpdateTextures();
}

/// Calculate the minimum and maximum values for the scalar field and set the low and high values if specified
void CalculateMinMax() {
    size_t N = pow(2, resolution);                                          // store the resolution of the field slices
    size_t N2 = N * N;

    real_max = S_xy[0].real();
    real_min = S_xy[0].real();
    imag_max = S_xy[0].imag();
    imag_min = S_xy[0].imag();

    for (size_t i = 0; i < N2; i++) {
        if (S_xy[i].real() > real_max) real_max = S_xy[i].real();
        if (S_xy[i].real() < real_min) real_min = S_xy[i].real();
        if (S_xy[i].imag() > imag_max) imag_max = S_xy[i].imag();
        if (S_xy[i].imag() < imag_min) imag_min = S_xy[i].imag();
        if (S_yz[i].real() > real_max) real_max = S_yz[i].real();
        if (S_yz[i].real() < real_min) real_min = S_yz[i].real();
        if (S_yz[i].imag() > imag_max) imag_max = S_yz[i].imag();
        if (S_yz[i].imag() < imag_min) imag_min = S_yz[i].imag();
        if (S_xz[i].real() > real_max) real_max = S_xz[i].real();
        if (S_xz[i].real() < real_min) real_min = S_xz[i].real();
        if (S_xz[i].imag() > imag_max) imag_max = S_xz[i].imag();
        if (S_xz[i].imag() < imag_min) imag_min = S_xz[i].imag();
    }
    if(!fix_low_high){
        real_low = real_min;
        real_high = real_max;
        imag_low = imag_min;
        imag_high = imag_max;
    }
}

/// Selects scalar values for the field slices based on user input
void EvaluateScalarSlices() {
    auto start = std::chrono::steady_clock::now();

    size_t N = pow(2, resolution);                                          // store the resolution of the field slices
    size_t N2 = N * N;

    // X-Y Scalar Evaluation
    for (size_t i = 0; i < N2; i++) {
        if (display_mode == DisplayMode::X)
            S_xy[i] = E_xy[i][0];
        if (display_mode == DisplayMode::Y)
            S_xy[i] = E_xy[i][1];
        if (display_mode == DisplayMode::Z)
            S_xy[i] = E_xy[i][2];
        if (display_mode == DisplayMode::Intensity)
            S_xy[i] = E_xy[i][0] * std::conj(E_xy[i][0]) +
                      E_xy[i][1] * std::conj(E_xy[i][1]) +
                      E_xy[i][2] * std::conj(E_xy[i][2]);
    }

    // Y-Z Scalar Evaluation
    for (size_t i = 0; i < N2; i++) {
        if (display_mode == DisplayMode::X)
            S_yz[i] = E_yz[i][0];
        if (display_mode == DisplayMode::Y)
            S_yz[i] = E_yz[i][1];
        if (display_mode == DisplayMode::Z)
            S_yz[i] = E_yz[i][2];
        if (display_mode == DisplayMode::Intensity)
            S_yz[i] = E_yz[i][0] * std::conj(E_yz[i][0]) +
                      E_yz[i][1] * std::conj(E_yz[i][1]) +
                      E_yz[i][2] * std::conj(E_yz[i][2]);
    }

    // X-Z Scalar Evaluation
    for (size_t i = 0; i < N2; i++) {
        if (display_mode == DisplayMode::X)
            S_xz[i] = E_xz[i][0];
        if (display_mode == DisplayMode::Y)
            S_xz[i] = E_xz[i][1];
        if (display_mode == DisplayMode::Z)
            S_xz[i] = E_xz[i][2];
        if (display_mode == DisplayMode::Intensity)
            S_xz[i] = E_xz[i][0] * std::conj(E_xz[i][0]) +
                      E_xz[i][1] * std::conj(E_xz[i][1]) +
                      E_xz[i][2] * std::conj(E_xz[i][2]);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_EvaluateScalarSlices = duration.count();
    CalculateMinMax();
    EvaluateColorSlices();
}

void EvaluateVectorSlices() {
    auto start = std::chrono::steady_clock::now();
    unsigned int N = pow(2, resolution);                         // get the resolution of the field N
    float d = extent / (N - 1);                                             // calculate the step size in cartesian coordinates
    float x, y, z;
    float x_start = center[0] - extent / 2;
    float y_start = center[1] - extent / 2;
    float z_start = center[2] - extent / 2;
                                                                            // stores the plane index of the current pixel

    // X-Y Vector Evaluation
    //z = plane_position[2];                                                  // all coordinates in this plane have the same z value
    if(in_device >= 0)
        gpu_cw_evaluate((thrust::complex<float>*)E_xy, (thrust::complex<float>*)E_xz, (thrust::complex<float>*)E_yz,
            x_start, y_start, z_start, plane_position[0], plane_position[1], plane_position[2], d, N, in_device);
    else {
        cpu_cw_evaluate_xy(E_xy, x_start, y_start, plane_position[2], d, N);
        cpu_cw_evaluate_xz(E_xz, x_start, z_start, plane_position[1], d, N);
        cpu_cw_evaluate_yz(E_yz, y_start, z_start, plane_position[0], d, N);
    }
    

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_EvaluateVectorSlices = duration.count();
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
        if(ImGui::InputScalar("Resolution = ", ImGuiDataType_U32, &resolution, &res_step, &res_step)){
            AllocateImageArrays();
            EvaluateVectorSlices();
        }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("%d x %d", (int)pow(2, resolution), (int)pow(2, resolution));

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
    io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", ui_scale * 16.0f);

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

void InitCuda(){
    // Initialize CUDA
    int nDevices;
    cudaError error = cudaGetDeviceCount(&nDevices);
   
    if (error != cudaSuccess || nDevices == 0) in_device = -1;                                                 // if there is an error getting device information, assume there are no devices

    if (in_device >= 0 && in_device < nDevices) {
        gpu_initialize();
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
    }

}
static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int argc, char** argv)
{

    boost::program_options::options_description desc("Allowed options");
	desc.add_options()
        ("input", boost::program_options::value<std::string>(&in_filename)->default_value("a.cw"), "output filename for the coupled wave structure")
		("help", "produce help message")
        ("cuda,c", boost::program_options::value<int>(&in_device)->default_value(0), "cuda device number (-1 is CPU-only)")
		("verbose,v", "produce verbose output")
		;
	boost::program_options::variables_map vm;

    boost::program_options::positional_options_description p;
    p.add("input", -1);
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    //boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
	
    boost::program_options::notify(vm); 
  

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}
    if(vm.count("verbose")){
        verbose = true;
    }
    
    if(!vm.count("input")){
        throw "ERROR: no input file specified";
    }
    AllocateImageArrays();                                              // allocate space to store the evaluated fields

    auto start = std::chrono::steady_clock::now();
    cw.load(in_filename);                                                  // load the coupled wave data
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end-start;
    t_LoadData = duration.count();

    cw_allocate(&cw);
    cw_unpack(&cw);

    InitCuda();                                                         // initialize CUDA

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // GL 3.0 + GLSL 130
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create window with graphics context
    std::string window_title = "ScatterView - " + in_filename;
    window = glfwCreateWindow(1600, 1200, window_title.c_str(), NULL, NULL);
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

    

    Material_xy.LoadShader("simple_texturemap.shader");          // create a material    
    Material_xz.LoadShader("simple_texturemap.shader");
    Material_yz.LoadShader("simple_texturemap.shader");

    EvaluateVectorSlices();    

    SliceGeometry = tira::glGenerateRectangle<float>();
    

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