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

    /**Number of qubits*/
    constexpr int dim = 5;

    /**Number of signal inputs*/
    constexpr int M = 1000;
    
    set_system_dimension(dim);
    set_data_output_path(DATA_OUTPUT_PATH);

    /**Random real number generator*/
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> distr(0, 1);

    //clear_data_folder();
    vector<SparseMatrix<double>*> sigma = generate_all_sigma();
    time_point startTime = high_resolution_clock::now();

    /*Grid specs*/
    std::pair<double, double> h_range(0.1, 1.0);
    std::pair<double, double> dt_range(1.0, 10.0);
    int h_num_samples = 8;
    int dt_num_samples = 8;
    double h_step = (h_range.second - h_range.first) / (h_num_samples - 1);
    double dt_step = (dt_range.second - dt_range.first) / (dt_num_samples - 1);

    for (int t = 5; t < 10; ++t) {
        cout << "Generating " << t << endl;

        for (double dt_pos = dt_range.first; dt_pos <= dt_range.second; dt_pos += dt_step) {

            #pragma omp parallel for default(none) shared(cout, h_range, dt_range, h_step, dt_step, dt_pos)
            for (int j = 0; j < h_num_samples; ++j) {
                cout << "Generating " << dt_pos << "," << h_step*j + h_range.first << endl;
                std::ostringstream filename;
                filename << "run_tau" << t << "_" << "dt" << dt_pos << "_" << "h" << h_step*j + h_range.first;
                
                for (int i = 0; i < 20; ++i) {
                    /**Random input vector initialization*/
                    VectorXd input(M + t);
                    for (int i = 0; i < M + t; ++i) {
                        input[i] = distr(gen);
                    }

                    VectorXd output(M);
                    for (int i = 0; i < M; ++i) {
                        output[i] = input[i];
                    }

                    MatrixXd* hamiltonian = ising_hamiltonian(&VectorXd(dim).setConstant(h_range.first + j*h_step));

                    /**Time evolution operator exp(-iHdt)*/
                    MatrixXcd* exp_s = time_evolution_operator(hamiltonian, dt_pos);
                    /**Its inverse (which is just the conjugate transpose)*/
                    MatrixXcd* exp_d = new MatrixXcd(exp_s->rows(), exp_s->cols());
                    *exp_d << exp_s->transpose().conjugate();

                    /**Density matrix corresponding to the state where all qubits are in the |0> state*/
                    MatrixXcd* rho = new MatrixXcd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
                    rho->setZero();
                    (*rho)(0, 0) = 1;

                    wash_out(&rho, M, exp_s, exp_d);

                    MatrixXd* training_measurements = measure_output(&rho, sigma, &input, &output, exp_s, exp_d, t);

                    appendMatrixToCSV(training_measurements, filename.str());

                    /*TESTING PHASE*/

                    for (int i = 0; i < M + t; ++i) {
                        input[i] = distr(gen);
                    }

                    for (int i = 0; i < M; ++i) {
                        output[i] = input[i];
                    }

                    wash_out(&rho, M, exp_s, exp_d);

                    MatrixXd* test_measurements = measure_output(&rho, sigma, &input, &output, exp_s, exp_d, t);

                    appendMatrixToCSV(test_measurements, filename.str());

                    delete training_measurements;
                    delete test_measurements;
                    delete hamiltonian;
                    delete exp_s;
                    delete exp_d;
                    delete rho;
                }
                filename.str("");
                filename.clear();
            }
        }

    }


    time_point endTime = std::chrono::high_resolution_clock::now();
    nanoseconds duration = endTime - startTime;
    long long ms = duration_cast<milliseconds>(duration).count();
    cout << "Elapsed: " << ms << " ms" << endl;


    return 0;
}