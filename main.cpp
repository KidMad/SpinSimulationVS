#include<iostream>
#include<numeric>
#include <Eigen/Dense>
#include <chrono>
#include <random>
#include <fstream>
#include "tools.h"


using Eigen::MatrixXcd;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;
using std::cout;
using std::endl;
using std::vector;
using std::string;

using namespace tools;

constexpr char* OPERATORS_PATH = "../../../operators";
constexpr char* DATA_OUTPUT_PATH = "E:\\Desktop\\Edu\\University\\3 Anno\\Tesi\\AnalisiDati\\data\\";

int main(){

    /**Number of qubits*/
    int dim = 7;
    /**Timestep*/
    int tau = 3;
    double dt = 20;

    /**Number of signal inputs*/
    constexpr int M = 1000;
    
    /**Creating operators*/
    set_sigma_operators_path(OPERATORS_PATH);
    generate_mb_sigma_operators(dim);
    
    set_data_output_path(DATA_OUTPUT_PATH);

    /**Random real number generator*/
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> distr(0, 1);

    /**Random input vector initialization*/
    VectorXd inputs(M);
    for (int i = 0; i < M; ++i) {
        inputs[i] = (distr(gen));
    }
    cout << "Initialized inputs" << endl;

    VectorXd outputs(M);
    outputs.setConstant(0);

    for (int i = tau; i < M; ++i) {
        outputs[i] = inputs[i-tau];
    }
    cout << "Initialized outputs" << endl;

    /**Ising hamiltonian initialized with the desired parameters*/
    auto hamiltonian = ising_hamiltonian(0.1, 1, dim);
    cout << "Initialized hamiltonian" << endl;

    /**Density matrix corresponding to the state where all qubits are in the |0> state*/
    auto rho = new MatrixXcd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
    rho->setZero();
    (*rho)(0, 0) = 1;
    cout << "Initialized density matrix" << endl;

    /**We decompose the hamiltonian to build the time evolution operators through exponentiation*/
    SelfAdjointEigenSolver<MatrixXcd> es(*hamiltonian);
    cout << "Decomposed hamiltonian" << endl;

    /**H = U*D*U_inv*/
    /*We are using the order given by the algorithm's solution, it doesn't really matter.**/
    auto U = new MatrixXcd(hamiltonian->rows(), hamiltonian->cols());
    (*U) << es.eigenvectors();
    /*We exploit the DiagonalMatrix class to avoid useless calculations.**/
    auto D = new Eigen::DiagonalMatrix<std::complex<double>, Eigen::Dynamic>(hamiltonian->rows());
    D->diagonal() << es.eigenvalues();

    /*We explicitly calculate the eigenvectors' matrix inverse.**/
    auto U_inv = new MatrixXcd(hamiltonian->rows(), hamiltonian->cols());
    *U_inv << U->inverse();

    /**Time evolution operator exp(-iHdt)*/
    auto exp_s = time_evolution_operator(dt, D, U, U_inv);
    /**Its inverse (which is just the conjugate transpose)*/
    auto exp_d = new MatrixXcd(exp_s->rows(), exp_s->cols());
    *exp_d << exp_s->transpose().conjugate();

    cout << "Built time evolution operators" << endl;

    /**Just to save memory since we don't need them anymore (hopefully)*/
    delete U;
    delete D;
    delete U_inv;

    int n_meas = 3 * dim + 3 * dim * (dim - 1) / 2;
    /**We are doing 3*dim + 3*(dim-1)*dim/2 measurements per sample i.e. we have 3*dim + 3*(dim-1)*dim/2 columns and M rows*/
    MatrixXd measurements(M, n_meas);

    /*We start from the initial matrix rho**/
    MatrixXcd* rho_ev = rho;

    auto startTime = std::chrono::high_resolution_clock::now();
    cout << "Saving measurements:" << endl;

    auto all = load_all_sigma(dim);

    /**Here starts the routine*/
    for (int s = 0; s < M; ++s) {
        MatrixXcd* tmp = rho_ev;

        rho_ev = reset(inputs[s], rho_ev);
        (*rho_ev) = *exp_s * *rho_ev * *exp_d;

        average_single(rho_ev, all, measurements, s);

        average_double(rho_ev, all, measurements, s);

        delete tmp;
    }

    cout << endl;
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = endTime - startTime;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

    cout << "Calculated measurements in " << static_cast<double>(us.count()) / 1000.0 << "ms" << endl;
    
    
    //Eigen::JacobiSVD <MatrixXd> svd(measurements, Eigen::ComputeThinU | Eigen::ComputeThinV);
    //cout << "Generated measurements' matrix SVD" << endl;
    //VectorXd weights = svd.solve(outputs);

    //cout << "Calculated weights" << endl;

    //clear_data_folder();
    //exportMatrixToCSV(&measurements, "measurements");
    //exportVectorToCSV(&inputs, "inputs");
    //exportVectorToCSV(&outputs, "outputs");

    
    return 0;
}