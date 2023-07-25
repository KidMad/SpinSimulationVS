#include<iostream>
#include <chrono>
#include "tools.h"

using std::chrono::high_resolution_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::nanoseconds;
using std::chrono::milliseconds;

using Eigen::MatrixXcd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;

using std::cout;
using std::endl;
using std::vector;
using std::string;

using namespace tools;

constexpr char* OPERATORS_PATH = "./operators/";
constexpr char* DATA_OUTPUT_PATH = "../../../data/";

int main() {

    time_point startTime = high_resolution_clock::now();

    /**Number of qubits*/
    constexpr int dim = 5;

    /**Timestep*/
    int tau = 2;
    double dt = 20;

    /**Number of signal inputs*/
    constexpr int M = 1000;
    
    set_system_dimension(dim);
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
    MatrixXd* hamiltonian = ising_hamiltonian(&VectorXd(dim).setConstant(1));
    cout << "DONE" << endl;

    cout << "Generating time evolution operators...";
    /**Time evolution operator exp(-iHdt)*/
    MatrixXcd* exp_s = time_evolution_operator(hamiltonian, dt);
    /**Its inverse (which is just the conjugate transpose)*/
    MatrixXcd* exp_d = new MatrixXcd(exp_s->rows(), exp_s->cols());
    *exp_d << exp_s->transpose().conjugate();
    cout << "DONE" << endl;

    cout << "Generating density matrix...";
    /**Density matrix corresponding to the state where all qubits are in the |0> state*/
    MatrixXcd* rho = new MatrixXcd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
    rho->setZero();
    (*rho)(0, 0) = 1;
    cout << "DONE" << endl;


    cout << "Washing out density matrix...";
    wash_out(&rho, M, exp_s, exp_d);
    cout << "DONE" << endl;

    cout << "Generating Pauli operators...";
    vector<SparseMatrix<double>*> sigma = generate_all_sigma();
    cout << "DONE" << endl;

    cout << "Measuring...";
    MatrixXd* training_measurements = measure_output(&rho, sigma, &inputs, exp_s, exp_d, tau);
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

    cout << "Measuring...";
    MatrixXd* test_measurements = measure_output(&rho, sigma, &inputs, exp_s, exp_d, tau);
    cout << "DONE" << endl;

    cout << "Exporting testing data...";
    exportMatrixToCSV(test_measurements, "testing_measurements");
    exportVectorToCSV(&inputs, "testing_inputs");
    exportVectorToCSV(&outputs, "testing_outputs");
    cout << "DONE" << endl;

    cout << endl;

    time_point endTime = std::chrono::high_resolution_clock::now();
    nanoseconds duration = endTime - startTime;
    long long ms = duration_cast<milliseconds>(duration).count();
    cout << "Finished. Elapsed time: " << ms << "ms" << endl;


    return 0;
}