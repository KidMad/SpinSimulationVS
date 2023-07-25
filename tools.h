
#ifndef SPINSIMULATION_TOOLS_H
#define SPINSIMULATION_TOOLS_H

#endif //SPINSIMULATION_TOOLS_H

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdarg>
#include <random>
#include <sstream>

using Eigen::MatrixXcd;
using Eigen::SparseMatrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixX;

namespace tools {
    extern int dim;
    extern std::string sigma_operators_path;
    extern std::string data_output_path;

    void set_system_dimension(int dim);
    void set_sigma_operators_path(const std::string& new_path);
    void set_data_output_path(const std::string& new_path);

    enum Pauli {
        X = 0,
        Y = 1,
        Z = 2,
    };

    template<typename T>
    MatrixX<T>* kronecker_product(const std::vector<MatrixX<T>>& matrices);

    template<typename T>
    MatrixX<T>* kronecker_product(std::initializer_list<MatrixX<T>*> matrices);

    /**Pauli matrices*/
    MatrixXd* sigmaX(int nth);
    MatrixXd* sigmaY(int nth);
    MatrixXd* sigmaZ(int nth);

    SparseMatrix<double>* sparseSigmaX(int nth);
    SparseMatrix<double>* sparseSigmaY(int nth);
    SparseMatrix<double>* sparseSigmaZ(int nth);

    void generate_mb_sigma_operators(const std::string& path=tools::sigma_operators_path);
    MatrixXd* load_sigma(int nth_qubit, const Pauli which, const std::string& path=tools::sigma_operators_path);
    std::vector<MatrixXd*> load_all_sigma(const int dim, const std::string& path = tools::sigma_operators_path);
    std::vector<SparseMatrix<double>*> generate_all_sigma();


    MatrixXd* ising_hamiltonian(const VectorXd* h, const double J = 1.0);
    void reset(MatrixXcd** rho, double signal_k);
    void wash_out(MatrixXcd** rho, int input_size, MatrixXcd* time_ev_op_s, MatrixXcd* time_ev_op_d);
    MatrixXcd* time_evolution_operator(MatrixXd* hamiltonian, const double dt);


    void average_single(MatrixXcd* rho, std::vector<SparseMatrix<double>*> sigma, MatrixXd* output, int sample, int colshift=0);
    void average_double(MatrixXcd* rho, std::vector<SparseMatrix<double>*> sigma, MatrixXd* output, int sample, int colshift = 0);
    MatrixXd* measure_output(MatrixXcd** rho, std::vector<SparseMatrix<double>*> sigma, VectorXd* input, VectorXd* output, MatrixXcd* time_ev_op_s, MatrixXcd* time_ev_op_d, int tau);


    void clear_data_folder();
    void exportMatrixToCSV(MatrixXd* data, const std::string& filename);
    void exportVectorToCSV(Eigen::VectorXd* data, const std::string& filename);

    void appendMatrixToCSV(MatrixXd* data, const std::string& filename);
}