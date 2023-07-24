#include<iostream>
#include <chrono>
#include "tools.h"


using Eigen::MatrixXcd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;
using std::cout;
using std::endl;
using std::vector;
using std::string;

using namespace tools;

constexpr char* OPERATORS_PATH = "./operators/";
constexpr char* DATA_OUTPUT_PATH = "../../../data/";

int main() {
    auto startTime = std::chrono::high_resolution_clock::now();

    /**Number of qubits*/
    constexpr int dim = 5;

    /**Timestep*/
    int tau = 3;
    double dt = 20;

    /**Number of signal inputs*/
    constexpr int M = 1000;
    
    set_system_dimension(dim);
    set_sigma_operators_path(OPERATORS_PATH);
    set_data_output_path(DATA_OUTPUT_PATH);

    /**Random real number generator*/
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> distr(0, 1);

    cout << "Initializing random input signal sequence...";
    /**Random input vector initialization*/
    VectorXd inputs(M + tau);
    for (int i = 0; i < M + tau; ++i) {
        inputs[i] = distr(gen);
    }
    cout << "DONE" << endl;


    cout << "Initializing delayed output sequence from input signal...";
    VectorXd outputs(M);
    for (int i = 0; i < M; ++i) {
        outputs[i] = inputs[i];
    }
    cout << "DONE" << endl;

    cout << "Generating random Ising hamiltonian...";
    /**Ising hamiltonian initialized with the desired parameters*/
    std::uniform_real_distribution<double>J_dist(-1, 1);
    auto hamiltonian = ising_hamiltonian(0.1, J_dist(gen));
    cout << "DONE" << endl;

    cout << "Decomposing hamiltonian...";
    /**We decompose the hamiltonian to build the time evolution operators through exponentiation*/
    SelfAdjointEigenSolver<MatrixXd> es(*hamiltonian);

    /**H = U*D*U_inv*/
    /*We are using the order given by the algorithm's solution, it doesn't really matter.**/
    auto U = new MatrixXd(hamiltonian->rows(), hamiltonian->cols());
    (*U) << es.eigenvectors();

    /*We exploit the DiagonalMatrix class to avoid useless calculations.**/
    auto D = new Eigen::DiagonalMatrix<double, Eigen::Dynamic>(hamiltonian->rows());
    D->diagonal() << es.eigenvalues();

    /*We explicitly calculate the eigenvectors' matrix inverse.**/
    auto U_inv = new MatrixXd(hamiltonian->rows(), hamiltonian->cols());
    *U_inv << U->transpose();

    cout << "DONE" << endl;

    cout << "Generating time evolution operators...";
    /**Time evolution operator exp(-iHdt)*/
    auto exp_s = time_evolution_operator(dt, D, U, U_inv);
    /**Its inverse (which is just the conjugate transpose)*/
    auto exp_d = new MatrixXcd(exp_s->rows(), exp_s->cols());
    *exp_d << exp_s->transpose().conjugate();
    cout << "DONE" << endl;

    /**Just to save memory since we don't need them anymore (hopefully)*/
    delete U;
    delete D;
    delete U_inv;

    cout << "Generating density matrix...";
    /**Density matrix corresponding to the state where all qubits are in the |0> state*/
    auto rho = new MatrixXcd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
    rho->setZero();
    (*rho)(0, 0) = 1;
    cout << "DONE" << endl;
    cout << "Washing out density matrix...";
    wash_out(&rho, M, exp_s, exp_d);
    cout << "DONE" << endl;

    cout << "Generating Pauli operators...";
    auto sigma = generate_all_sigma();
    cout << "DONE" << endl;

    cout << "Injecting initial signal into system...";
    inject_initial_signal(&rho, &inputs, exp_s, exp_d, tau);
    cout << "DONE" << endl;

    cout << "Measuring...";
    auto training_measurements = measure_output(&rho, sigma, &inputs, exp_s, exp_d, tau);
    cout << "DONE" << endl;

    cout << "Exporting training data...";
    clear_data_folder();
    exportMatrixToCSV(training_measurements, "training_measurements");
    exportVectorToCSV(&inputs, "training_inputs");
    exportVectorToCSV(&outputs, "training_outputs");
    cout << "DONE" << endl;

    
    /*TESTING PHASE*/

    cout << "Creating test input...";
    for (int i = 0; i < M + tau; ++i) {
        inputs[i] = distr(gen);
    }
    cout << "DONE" << endl;

    cout << "Creating output for comparison...";
    for (int i = 0; i < M; ++i) {
        outputs[i] = inputs[i];
    }
    cout << "DONE" << endl;


    cout << "Washing out density matrix...";
    wash_out(&rho, M, exp_s, exp_d);
    cout << "DONE" << endl;

    cout << "Injecting initial signal...";
    inject_initial_signal(&rho, &inputs, exp_s, exp_d, tau);
    cout << "DONE" << endl;

    cout << "Measuring...";
    auto test_measurements = measure_output(&rho, sigma, &inputs, exp_s, exp_d, tau);
    cout << "DONE" << endl;


    cout << "Exporting testing data...";
    exportMatrixToCSV(test_measurements, "testing_measurements");
    exportVectorToCSV(&inputs, "testing_inputs");
    exportVectorToCSV(&outputs, "testing_outputs");
    cout << "DONE" << endl;

    cout << endl;

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = endTime - startTime;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    cout << "Finished. Elapsed time: " << static_cast<double>(us.count()) / 1000.0 << "ms" << endl;


    return 0;
}