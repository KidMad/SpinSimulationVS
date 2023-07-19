
#ifndef SPINSIMULATION_TOOLS_H
#define SPINSIMULATION_TOOLS_H

#endif //SPINSIMULATION_TOOLS_H

#include <Eigen/Dense>

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::Matrix;

namespace tools {
    extern std::string sigma_operators_path;
    extern std::string data_output_path;
    extern enum Pauli;

    template<typename T>
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* kronecker_product(const std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& matrices);

    template<typename T>
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* kronecker_product(std::initializer_list<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>*> matrices);

    /**Pauli matrices*/
    MatrixXd* sigmaX(int nth, int dim);
    MatrixXd* sigmaY(int nth, int dim);
    MatrixXd* sigmaZ(int nth, int dim);

    MatrixXd* ising_hamiltonian(double H, double J, int dim);

    MatrixXcd* reset(double signal_k, MatrixXcd* rho);
    MatrixXcd* time_evolution_operator(const double dt, Eigen::DiagonalMatrix<double, Eigen::Dynamic>* D, MatrixXd* U, MatrixXd* U_inv);
    void average_single(MatrixXcd* rho, std::vector<MatrixXd*> all, MatrixXd& output, int sample);
    void average_double(MatrixXcd* rho, std::vector<MatrixXd*> all, MatrixXd& output, int sample);

    void exportMatrixToCSV(MatrixXd* data, const std::string& filename);
    void exportVectorToCSV(Eigen::VectorXd* data, const std::string& filename);
    void clear_data_folder();

    void generate_mb_sigma_operators(const int dim, const std::string& path=tools::sigma_operators_path);
    void set_sigma_operators_path(const std::string& new_path);
    void set_data_output_path(const std::string& new_path);
    MatrixXd* load_sigma(int nth_qubit, const Pauli which, const std::string& path=tools::sigma_operators_path);
    std::vector<MatrixXd*> load_all_sigma(const int dim, const std::string& path = tools::sigma_operators_path);
}