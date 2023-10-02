#include "cpuEvaluator.h"

int layers;
std::vector<float> z_layers;
std::vector<int> waves_begin;
std::vector<int> waves_end;
std::vector<UnpackedWave<float>> W;
unsigned int z_up;
unsigned int z_bo;

namespace {
    bool isHete;                                            // Visualize a heterogeneous sample
    int slices;                                             // Number of layers for the heterogeneous sample
    unsigned int M[2];                                      // Fourier coefs along x and y
    unsigned int _M;                                        // M[0] * M[1]
    double size[3];                                         // sample size along x, y, and z
    std::vector < std::vector< std::complex<double>>> NIf;  // Fourier coefficients of each sample layer
    std::complex<double> K;                                 // Wavenumber
    glm::vec<3, double > s;                                 // Wave propagation direction (centered)
    double X[2];                                            // Boundary coordinates for sample along x
    double Y[2];                                            // Boundary coordinates for sample along y
    unsigned int points;                                    // Number of pixels along each axis
    float extent;                                           // Size the the visualization canvas
    Eigen::ArrayXcd tmp_a;                                  // Temperary media to store array, matrix, or vector
    Eigen::MatrixXcd tmp_m;
    Eigen::VectorXcd tmp_v;
    std::vector<Eigen::VectorXcd> Beta;                     // Some physical property for the hete sample
    std::vector<Eigen::VectorXcd> Gamma;                    // Eigenvalues for the hete sample
    std::vector<Eigen::MatrixXcd> GG;                       // Eigenvectors for the hete sample
}
void cw_allocate(CoupledWaveStructure<double>* cw) {
    layers = cw->Layers.size();
    slices = cw->Slices.size();
    z_layers.resize(layers);

    isHete = cw->isHete;
    if (isHete) {
        M[0] = cw->M[0];
        M[1] = cw->M[1];
        _M = M[0] * M[1];
        size[0] = cw->size[0];
        size[1] = cw->size[1];
        size[2] = cw->size[2];

        NIf.resize(slices);
        for (int j = 0; j < slices; j++) {
            NIf[j].resize(_M);
            for (int i = 0; i < _M; i++) {
                NIf[j][i] = cw->NIf[j][i];
            }
        }

        glm::vec<3, std::complex<float> > k = cw->Pi[M[1] / 2 * M[0] + M[0] / 2].getK();
        K = sqrt(pow(k[0], 2) + pow(k[1], 2) + pow(k[2], 2));
        Beta.resize(slices);
        Gamma.resize(slices);
        GG.resize(slices);
        for (size_t i = 0; i < slices; i++) {
            Beta[i] = Eigen::Map < Eigen::VectorXcd>(cw->Slices[i].beta.data(), 4 * _M);
            Gamma[i] = Eigen::Map< Eigen::VectorXcd>(cw->Slices[i].gamma.data(), 4 * _M);
            GG[i] = Eigen::Map< Eigen::MatrixXcd>(cw->Slices[i].gg.data(), 4 * _M, 4 * _M);
            GG[i].transposeInPlace();

        }
    }

    waves_begin.reserve(1000);
    waves_end.reserve(1000);
    waves_begin.push_back(0);
    int total_waves = cw->Pi.size();
    s = cw->Pi[M[1] / 2 * M[0] + M[0] / 2].getDirection();

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

unsigned int idx(unsigned int num) {
    float ratio = size[2] / (float)extent;
    float p = points* ratio;
    return unsigned int(float(num * slices) / p);
}

void EvaluateSample(std::vector <std::vector< Eigen::MatrixXcd>>& E, float* center, float Extent, unsigned int N) {
    // Visualization boundaries. Eg: extent=100, center=[50, 50, 0]. X=Y=[0, 100]
    extent = Extent;
    X[0] = center[0] - extent / 2.0;
    Y[0] = center[1] - extent / 2.0;
    X[1] = center[0] + extent / 2.0;
    Y[1] = center[1] + extent / 2.0;

    points = N;

    // Newly added on 09/18/2023
    unsigned int points_z = (z_layers[1] - z_layers[0]) / extent * (float)(points-1);
    if (Beta.size() > 1)
        points_z = Beta.size();
    z_up = unsigned int(((z_layers[0] - (extent / 2.0 - center[2])) * (float)(points-1) / (float)extent)) + 1;
    z_bo = z_up + points_z;

    E.resize(3);
    E[0].resize(points_z);
    E[1].resize(points_z);
    E[2].resize(points_z);

    Eigen::VectorXd p_series;
    Eigen::VectorXd q_series;
    p_series.setLinSpaced(M[0], -double(M[0] / 2), double((M[0] - 1) / 2));                         // M=3: p_series=[-1, 0, 1]. M=2: p_series=[-1, 0]
    q_series.setLinSpaced(M[1], -double(M[1] / 2), double((M[1] - 1) / 2));

    Eigen::VectorXd WQ = 2.0 * q_series * PI / size[1];
    Eigen::VectorXd UP = 2.0 * p_series * PI / size[0];
    std::vector<std::vector<Eigen::ArrayXcd>> Ef(3);
    Ef[0].resize(points_z);
    Ef[1].resize(points_z);
    Ef[2].resize(points_z);
    Eigen::ArrayXcd I;
    Eigen::ArrayXcd J;
    I.resize(_M);
    J.resize(_M);
    std::complex<double> S[2];
    S[0] = s[0];
    S[1] = s[1];

    for (int z = 0; z < points_z; z++) {
        unsigned int i = 0;     // For single-layered sample
        if (Beta.size() > 1)
            i = (unsigned int)z;
        Ef[0][z].resize(_M);
        Ef[1][z].resize(_M);
        Ef[2][z].resize(_M);
        Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> beta_even(Beta[i].data(), 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
        Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> beta_odd(Beta[i].data() + 1, 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
        Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> gamma_even(Gamma[i].data(), 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
        Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> gamma_odd(Gamma[i].data() + 1, 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));

        //Eigen::MatrixXcd phase_even = (std::complex<double>(0, 1) * K * (std::complex<double>)((z + z_up) * extent / (float)(points - 1) - extent / 2.0 - z_layers[0]) * gamma_even.array()).exp().matrix();
        //Eigen::MatrixXcd phase_odd = (std::complex<double>(0, 1) * K * (std::complex<double>)((z + z_up) * extent / (float)(points - 1) - extent / 2.0 - z_layers[1]) * gamma_odd.array()).exp().matrix();
        Eigen::MatrixXcd phase_even = (std::complex<double>(0, 1) * K * (std::complex<double>)((z ) * extent / (float)(points - 1)) * gamma_even.array()).exp().matrix();
        Eigen::MatrixXcd phase_odd = (std::complex<double>(0, 1) * K * (std::complex<double>)((z) * extent / (float)(points - 1) + z_layers[0] - z_layers[1]) * gamma_odd.array()).exp().matrix();

        if (Beta.size() > 1) {
            phase_even = (std::complex<double>(0, 1) * K * (std::complex<double>)(0) * gamma_even.array()).exp().matrix();
            phase_odd = (std::complex<double>(0, 1) * K * (std::complex<double>)(-extent / (float)(points - 1)) * gamma_odd.array()).exp().matrix();
        }

        for (int n = 0; n < _M; n++) {
            Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> gg_even(GG[i].data() + 4 * _M * n, 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
            Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> gg_odd(GG[i].data() + 4 * _M * n + 1, 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
            Ef[0][z][n] = (beta_even.array() * gg_even.array() * phase_even.array() + beta_odd.array() * gg_odd.array() * phase_odd.array()).sum();

            gg_even = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(GG[i].data() + 4 * _M * (_M * 1 + n), 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
            gg_odd = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(GG[i].data() + 4 * _M * (_M * 1 + n) + 1, 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
            Ef[1][z][n] = (beta_even.array() * gg_even.array() * phase_even.array() + beta_odd.array() * gg_odd.array() * phase_odd.array()).sum();

            gg_even = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(GG[i].data() + 4 * _M * (_M * 2 + n), 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
            gg_odd = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(GG[i].data() + 4 * _M * (_M * 2 + n) + 1, 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
            I[n] = (beta_even.array() * gg_even.array() * phase_even.array() + beta_odd.array() * gg_odd.array() * phase_odd.array()).sum();


            gg_even = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(GG[i].data() + 4 * _M * (_M * 3 + n), 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
            gg_odd = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(GG[i].data() + 4 * _M * (_M * 3 + n) + 1, 1, 2 * _M, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(2, 1));
            J[n] = (beta_even.array() * gg_even.array() * phase_even.array() + beta_odd.array() * gg_odd.array() * phase_odd.array()).sum();
        }

        for (int qi = 0; qi < M[1]; qi++) {
            for (int pi = 0; pi < M[0]; pi++) {
                Ef[2][z][qi * M[0] + pi] = 0;
                for (int qj = 0; qj < M[1]; qj++) {
                    int indR = int(qi - qj) % M[1];

                    std::complex<double> wq = std::complex<double>(WQ[qj]) + S[1] * K;
                    for (int pj = 0; pj < M[0]; pj++) {
                        int indC = int(pi - pj) % M[0];
                        std::complex<double> up = std::complex<double>(UP[pj]) + S[0] * K;
                        std::complex<double> ef2 = NIf[i][(indR + M[1] / 2) % M[1] * M[0] + (indC + M[0] / 2) % M[0]]
                            * (up * J[qj * M[0] + pj] - wq * I[qj * M[0] + pj]);
                        Ef[2][z][qi * M[0] + pi] += std::complex<double>(-1, 0) / K * ef2;
                    }
                }
            }

        }
        Eigen::VectorXd X_vec;
        Eigen::VectorXd Y_vec;
        X_vec.setLinSpaced(points, X[0], X[1]);
        Y_vec.setLinSpaced(points, Y[0], Y[1]);
        E[0][z] = fftw_ift2(Eigen::Map<Eigen::MatrixXcd>(Ef[0][z].data(), M[0], M[1]), X_vec, Y_vec, S, K);
        E[1][z] = fftw_ift2(Eigen::Map<Eigen::MatrixXcd>(Ef[1][z].data(), M[0], M[1]), X_vec, Y_vec, S, K);
        E[2][z] = fftw_ift2(Eigen::Map<Eigen::MatrixXcd>(Ef[2][z].data(), M[0], M[1]), X_vec, Y_vec, S, K);
    }
}

void cpu_cw_evaluate_sample(glm::vec<3, std::complex<float>>* E_xy, glm::vec<3, std::complex<float>>* E_xz, glm::vec<3, std::complex<float>>* E_yz,
    std::vector<std::vector<Eigen::MatrixXcd>> E,
    float x_start, float y_start, float z_start, float x, float y, float z, float d) {

    unsigned int pixel_x = unsigned int((x - x_start) / (float)extent * (float)points);
    unsigned int pixel_y = unsigned int((y - y_start) / (float)extent * (float)points);
    unsigned int pixel_z = unsigned int((z - z_start) / (float)extent * (float)points);
    if (pixel_x < 0)
        pixel_x = 0;
    if (pixel_y < 0)
        pixel_y = 0;
    if (pixel_z < 0)
        pixel_z = 0;
    if (pixel_x >= (unsigned int)points)
        pixel_x = points - 1;
    if (pixel_y >= (unsigned int)points)
        pixel_y = points - 1;
    if (pixel_z >= (unsigned int)points)
        pixel_z = points - 1;

    if (pixel_z >= z_up && pixel_z < z_bo)
        for (size_t j = 0; j < points; j++) {
            for (size_t i = 0; i < points; i++) {
                E_xy[j * points + i][0] += (std::complex<float>)E[0][pixel_z - z_up](i, j);
                E_xy[j * points + i][1] += (std::complex<float>)E[1][pixel_z - z_up](i, j);
                E_xy[j * points + i][2] += (std::complex<float>)E[2][pixel_z - z_up](i, j);
            }
        }

    for (size_t j = 0; j < points; j++) {
        for (size_t i = 0; i < points; i++) {
            if (j >= z_up && j < z_bo) {
                E_xz[j * points + i][0] += (std::complex<float>)E[0][j - z_up](i, pixel_y);
                E_xz[j * points + i][1] += (std::complex<float>)E[1][j - z_up](i, pixel_y);
                E_xz[j * points + i][2] += (std::complex<float>)E[2][j - z_up](i, pixel_y);
                                                                       
                E_yz[j * points + i][0] += (std::complex<float>)E[0][j - z_up](pixel_x, i);
                E_yz[j * points + i][1] += (std::complex<float>)E[1][j - z_up](pixel_x, i);
                E_yz[j * points + i][2] += (std::complex<float>)E[2][j - z_up](pixel_x, i);
            }
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