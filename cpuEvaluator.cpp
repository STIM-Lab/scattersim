#include "CoupledWaveStructure.h"
#include "cpuEvaluator.h"

int layers;
std::vector<float> z_layers;
std::vector<int> waves_begin;
std::vector<int> waves_end;
std::vector<UnpackedWave<float>> W;


void cw_allocate(CoupledWaveStructure<double>* cw){
    waves_begin.clear();
    waves_end.clear();
    layers = cw->Layers.size();
    z_layers.resize(layers);

    waves_begin.push_back(0);
    int total_waves = cw->Pi.size();
    
    for (int li = 0; li < cw->Layers.size(); li++) {
        z_layers[li] = cw->Layers[li].z;
        total_waves += cw->Layers[li].Pr.size();
        waves_end.push_back(total_waves);
        waves_begin.push_back(total_waves);
        total_waves += cw->Layers[li].Pt.size();
    }
    waves_end.push_back(total_waves);
    W.resize(total_waves);    
}

/// <summary>
/// Unpacks all plane waves in the Coupled Wave structure to arrays containing the E vector at 0 (E0) and the k vector.
/// These arrays perform all of the processing necessary to evaluate the plane wave at 0, making it easier to map to a fast
/// CPU and GPU calculation for visualization.
/// </summary>
/// <param name="cw"></param>
void cw_unpack(CoupledWaveStructure<double>* cw) {

    size_t idx = 0;
    glm::vec<3, std::complex<float> > E0(0, 0, 0);
    glm::vec<3, std::complex<float> > k(0, 0, 0);
    for (size_t pi = 0; pi < cw->Pi.size(); pi++) {
        E0 = cw->Pi[pi].getE0();
        k = cw->Pi[pi].getK();
        W[idx].E0[0] = E0[0];
        W[idx].E0[1] = E0[1];
        W[idx].E0[2] = E0[2];
        W[idx].k[0] = k[0];
        W[idx].k[1] = k[1];
        W[idx].k[2] = k[2];
        idx++;
    }

    for (size_t li = 0; li < cw->Layers.size(); li++) {
        for (size_t ri = 0; ri < cw->Layers[li].Pr.size(); ri++) {
            E0 = cw->Layers[li].Pr[ri].getE0();
            k = cw->Layers[li].Pr[ri].getK();
            W[idx].E0[0] = E0[0];
            W[idx].E0[1] = E0[1];
            W[idx].E0[2] = E0[2];
            W[idx].k[0] = k[0];
            W[idx].k[1] = k[1];
            W[idx].k[2] = k[2];
            idx++;
        }

        for (size_t ti = 0; ti < cw->Layers[li].Pt.size(); ti++) {
            E0 = cw->Layers[li].Pt[ti].getE0();
            k = cw->Layers[li].Pt[ti].getK();
            W[idx].E0[0] = E0[0];
            W[idx].E0[1] = E0[1];
            W[idx].E0[2] = E0[2];
            W[idx].k[0] = k[0];
            W[idx].k[1] = k[1];
            W[idx].k[2] = k[2];
            idx++;
        }
    }
}

void cpu_cw_evaluate_xy(glm::vec<3, std::complex<float>>* E_xy, 
    float x_start, float y_start,
    float z, float d, size_t N) {

    float x, y;

    // find the current layer
    size_t l = 0;
    for (size_t li = 0; li < layers; li++) {
        if (z >= z_layers[li]) {
            l = li + 1;
        }
    }

    size_t begin = waves_begin[l];
    size_t end = waves_end[l];

    
    std::complex<float> phase;
    std::complex<float> k_dot_r = 0;
    std::complex<float> i(0.0, 1.0);

    for (unsigned int iy = 0; iy < N; iy++) {
        y = y_start + iy * d;
        for (unsigned int ix = 0; ix < N; ix++) {
            x = x_start + ix * d;                            // calculate the x and y coordinates to be evaluated
            

            glm::vec<3, std::complex<float>> E(0, 0, 0);
            for (size_t cwi = begin; cwi < end; cwi++) {
                k_dot_r = x * W[cwi].k[0] + y * W[cwi].k[1] + z * W[cwi].k[2];
                phase = std::exp(i * k_dot_r);
                E[0] += W[cwi].E0[0] * phase;
                E[1] += W[cwi].E0[1] * phase;
                E[2] += W[cwi].E0[2] * phase;
            }

            E_xy[iy * N + ix] = E;
        }
    }
}

void cpu_cw_evaluate_yz(glm::vec<3, std::complex<float>>* E_yz,
    float y_start, float z_start,
    float x, float d, size_t N) {

    float y, z;
    std::complex<float> phase;
    std::complex<float> k_dot_r = 0;
    std::complex<float> i(0.0, 1.0);

    size_t l = 0;
    for (unsigned int iz = 0; iz < N; iz++) {
        z = z_start + iz * d;
        for (size_t li = 0; li < layers; li++) {
            if (z >= z_layers[li]) {
                l = li + 1;
            }
        }
        size_t begin = waves_begin[l];
        size_t end = waves_end[l];
        for (unsigned int iy = 0; iy < N; iy++) {
            y = y_start + iy * d;                            // calculate the x and y coordinates to be evaluated
            

            glm::vec<3, std::complex<float>> E(0, 0, 0);
            for (size_t cwi = begin; cwi < end; cwi++) {
                k_dot_r = x * W[cwi].k[0] + y * W[cwi].k[1] + z * W[cwi].k[2];
                phase = std::exp(i * k_dot_r);
                E[0] += W[cwi].E0[0] * phase;
                E[1] += W[cwi].E0[1] * phase;
                E[2] += W[cwi].E0[2] * phase;
            }

            E_yz[iz * N + iy] = E;
        }
    }
}

void cpu_cw_evaluate_xz(glm::vec<3, std::complex<float>>* E_xz,
    float x_start, float z_start,
    float y, float d, size_t N) {

    float x, z;
    std::complex<float> phase;
    std::complex<float> k_dot_r = 0;
    std::complex<float> i(0.0, 1.0);

    size_t l = 0;
    for (unsigned int iz = 0; iz < N; iz++) {
        z = z_start + iz * d;
        for (size_t li = 0; li < layers; li++) {
            if (z >= z_layers[li]) {
                l = li + 1;
            }
        }
        size_t begin = waves_begin[l];
        size_t end = waves_end[l];
        for (unsigned int ix = 0; ix < N; ix++) {
            x = x_start + ix * d;                            // calculate the x and y coordinates to be evaluated

            glm::vec<3, std::complex<double>> E(0, 0, 0);
            for (size_t cwi = begin; cwi < end; cwi++) {
                k_dot_r = x * W[cwi].k[0] + y * W[cwi].k[1] + z * W[cwi].k[2];
                phase = std::exp(i * k_dot_r);
                E[0] += W[cwi].E0[0] * phase;
                E[1] += W[cwi].E0[1] * phase;
                E[2] += W[cwi].E0[2] * phase;
            }

            E_xz[iz * N + ix] = E;
        }
    }
}