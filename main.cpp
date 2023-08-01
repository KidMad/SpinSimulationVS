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
    int h_num_samples = 32;
    int dt_num_samples = 32;
    double h_step = (h_range.second - h_range.first) / (h_num_samples - 1);
    double dt_step = (dt_range.second - dt_range.first) / (dt_num_samples - 1);

    int n_samples_mean = 20;

    for (int t = 11; t < 13; ++t) {
        cout << "Generating " << t << endl;

        std::ostringstream filename;
        filename << "run_tau" << t << "_" << "dt" << dt_range.first << "," << dt_range.second << "(" << dt_num_samples << ")" << "_" << "h" << h_range.first << "," << h_range.second << "(" << h_num_samples << ")";

        MatrixXd sampling(h_num_samples, dt_num_samples);

        for(int i = 0; i < h_num_samples; ++i){
            #pragma omp parallel for default(none) shared(cout, h_range, dt_range, h_step, dt_step)
            for (int j = 0; j < dt_num_samples; ++j) {
                cout << "Generating " << dt_range.first + j * dt_step << "," << h_range.first + i * h_step << endl;
                
                VectorXd capacities (n_samples_mean);

                for (int s = 0; s < n_samples_mean; ++s) {
                    /**Random input vector initialization*/
                    VectorXd training_input(M + t);
                    for (int i = 0; i < M + t; ++i) {
                        training_input[i] = distr(gen);
                    }

                    VectorXd training_output(M);
                    for (int i = 0; i < M; ++i) {
                        training_output[i] = training_input[i];
                    }

                    MatrixXd* hamiltonian = ising_hamiltonian(&VectorXd(dim).setConstant(h_range.first + i*h_step));

                    /**Time evolution operator exp(-iHdt)*/
                    MatrixXcd* exp_s = time_evolution_operator(hamiltonian, dt_range.first + j*dt_step);
                    /**Its inverse (which is just the conjugate transpose)*/
                    MatrixXcd* exp_d = new MatrixXcd(exp_s->rows(), exp_s->cols());
                    *exp_d << exp_s->transpose().conjugate();

                    /**Density matrix corresponding to the state where all qubits are in the |0> state*/
                    MatrixXcd* rho = new MatrixXcd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
                    rho->setZero();
                    (*rho)(0, 0) = 1;

                    wash_out(&rho, M, exp_s, exp_d);

                    MatrixXd* training_measurements = measure_output(&rho, sigma, &training_input, &training_output, exp_s, exp_d, t);

                    /*TESTING PHASE*/

                    VectorXd testing_input(M + t);
                    for (int i = 0; i < M + t; ++i) {
                        testing_input[i] = distr(gen);
                    }

                    VectorXd testing_output(M);
                    for (int i = 0; i < M; ++i) {
                        testing_output[i] = testing_input[i];
                    }

                    wash_out(&rho, M, exp_s, exp_d);

                    MatrixXd* test_measurements = measure_output(&rho, sigma, &testing_input, &testing_output, exp_s, exp_d, t);

                    /*Calculating weights using training data*/
                    VectorXd weights = training_measurements->bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(training_output);

                    /*Calculating predicted output using weights from training data and measurements from test data*/
                    VectorXd predicted_output = (*test_measurements) * weights;

                    double cov = ((predicted_output.array() - predicted_output.mean()) * (testing_output.array() - testing_output.mean())).sum() / (predicted_output.size() - 1);
                    double pred_var = (predicted_output.array() - predicted_output.mean()).square().sum() / (predicted_output.size() - 1);
                    double test_var = (testing_output.array() - testing_output.mean()).square().sum() / (testing_output.size() - 1);

                    capacities[s] = (cov * cov) / (pred_var * test_var);

                    delete training_measurements;
                    delete test_measurements;
                    delete hamiltonian;
                    delete exp_s;
                    delete exp_d;
                    delete rho;
                }

                sampling(i, j) = capacities.mean();

                //filename.str("");
                //filename.clear();
            }
        }
        exportMatrixToCSV(&sampling, filename.str());
    }

    time_point endTime = std::chrono::high_resolution_clock::now();
    nanoseconds duration = endTime - startTime;
    long long ms = duration_cast<milliseconds>(duration).count();
    cout << "Elapsed: " << ms << " ms" << endl;


    return 0;
}