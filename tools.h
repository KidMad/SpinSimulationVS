
#ifndef SPINSIMULATION_TOOLS_H
#define SPINSIMULATION_TOOLS_H

#endif //SPINSIMULATION_TOOLS_H

#include <Eigen/Dense>

using Eigen::MatrixXcd;
using Eigen::MatrixXd;

namespace tools {
    extern std::string sigma_operators_path;
    extern std::string data_output_path;

    MatrixXcd* kronecker_product(const std::vector<MatrixXcd>& matrices);
    MatrixXcd* kronecker_product(std::initializer_list<MatrixXcd*> matrices);

    /**Pauli matrices*/
    MatrixXcd* sigmaX(int nth, int dim);
    MatrixXcd* sigmaY(int nth, int dim);
    MatrixXcd* sigmaZ(int nth, int dim);

    MatrixXcd* ising_hamiltonian(double H, double J, int dim);

    MatrixXcd* reset(double signal_k, MatrixXcd* rho);
    MatrixXcd* time_evolution_operator(const double dt, Eigen::DiagonalMatrix<std::complex<double>, Eigen::Dynamic>* D, MatrixXcd* U, MatrixXcd* U_inv);
    void average_single(MatrixXcd* rho, std::vector<MatrixXcd*> all, MatrixXd& output, int sample);
    void average_double(MatrixXcd* rho, std::vector<MatrixXcd*> all, MatrixXd& output, int sample);

    void exportMatrixToCSV(MatrixXd* data, const std::string& filename);
    void exportVectorToCSV(Eigen::VectorXd* data, const std::string& filename);
    void clear_data_folder();

    void generate_mb_sigma_operators(const int dim, const std::string& path=tools::sigma_operators_path);
    void set_sigma_operators_path(const std::string& new_path);
    void set_data_output_path(const std::string& new_path);
    MatrixXcd* load_sigma(int nth_qubit, const std::string& which, const std::string& path=tools::sigma_operators_path);
    std::vector<MatrixXcd*> load_all_sigma(const int dim, const std::string& path = tools::sigma_operators_path);
}