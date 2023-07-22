
#ifndef SPINSIMULATION_TOOLS_H
#define SPINSIMULATION_TOOLS_H

#endif //SPINSIMULATION_TOOLS_H

#include <Eigen/Dense>

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix;

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
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* kronecker_product(const std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& matrices);

    template<typename T>
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* kronecker_product(std::initializer_list<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>*> matrices);

    /**Pauli matrices*/
    MatrixXd* sigmaX(int nth);
    MatrixXd* sigmaY(int nth);
    MatrixXd* sigmaZ(int nth);
    void generate_mb_sigma_operators(const std::string& path=tools::sigma_operators_path);
    MatrixXd* load_sigma(int nth_qubit, const Pauli which, const std::string& path=tools::sigma_operators_path);
    std::vector<MatrixXd*> load_all_sigma(const int dim, const std::string& path = tools::sigma_operators_path);
    std::vector<MatrixXd*> generate_all_sigma();


    MatrixXd* ising_hamiltonian(double H, double J, int dim);
    void reset(MatrixXcd** rho, double signal_k);
    void wash_out(MatrixXcd** rho, int input_size, MatrixXcd* time_ev_op_s, MatrixXcd* time_ev_op_d);
    MatrixXcd* time_evolution_operator(const double dt, Eigen::DiagonalMatrix<double, Eigen::Dynamic>* D, MatrixXd* U, MatrixXd* U_inv);

    void inject_initial_signal(MatrixXcd** rho, VectorXd* signal, MatrixXcd* time_ev_op_s, MatrixXcd* time_ev_op_d, int tau);
    void average_single(MatrixXcd* rho, std::vector<MatrixXd*> sigma, MatrixXd* output, int sample);
    void average_double(MatrixXcd* rho, std::vector<MatrixXd*> sigma, MatrixXd* output, int sample);
    MatrixXd* measure_output(MatrixXcd** rho, std::vector<MatrixXd*> sigma, VectorXd* signal, MatrixXcd* time_ev_op_s, MatrixXcd* time_ev_op_d, int tau);


    void clear_data_folder();
    void exportMatrixToCSV(MatrixXd* data, const std::string& filename);
    void exportVectorToCSV(Eigen::VectorXd* data, const std::string& filename);
}