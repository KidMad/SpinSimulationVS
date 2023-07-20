#include "tools.h"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdarg>
#include <sstream>

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix;
using std::complex;
using std::floor;
using std::cout;
using std::endl;

std::string tools::sigma_operators_path;
std::string tools::data_output_path;

void tools::set_sigma_operators_path(const std::string& new_path) {
    tools::sigma_operators_path = new_path;
}
void tools::set_data_output_path(const std::string& new_path) {
    tools::data_output_path = new_path;
}

enum tools::Pauli {
    X = 0,
    Y = 1,
    Z = 2,
};

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* tools::kronecker_product(const std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& matrices);
template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* tools::kronecker_product(const std::initializer_list<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>*> matrices) {

    long long int rows_dim = 1;
    long long int cols_dim = 1;

    for (int i = 0; i < matrices.size(); ++i) {
        rows_dim *= (*(matrices.begin() + i))->rows();
        cols_dim *= (*(matrices.begin() + i))->cols();
    }

    auto result = new Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(rows_dim, cols_dim);

#pragma omp parallel for default(none) shared(cout, rows_dim, cols_dim, matrices, result)
    for (int i = 0; i < rows_dim; ++i) {

        for (int j = 0; j < cols_dim; ++j) {

            /**It starts by calculating the first factor from the last matrix*/
            T value = (**(matrices.end() - 1))(i % (**(matrices.end() - 1)).rows(), j % (**(matrices.end() - 1)).cols());

            /**We start from i and j*/
            long long prev_row_index_division = i;
            long long prev_col_index_division = j;

            long long prev_matrix_rows_dim = (*(matrices.end() - 1))->rows();
            long long prev_matrix_cols_dim = (*(matrices.end() - 1))->cols();

            /**We go from the N - 1 matrix to the 2 matrix*/
            for (int l = 1; l < matrices.size() - 1; ++l) {
                const Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m = **(matrices.end() - 1 - l);

                auto new_row_index_division = static_cast<long long>(floor(prev_row_index_division / prev_matrix_rows_dim));
                auto new_col_index_division = static_cast<long long>(floor(prev_col_index_division / prev_matrix_cols_dim));
                value *= m(
                    new_row_index_division % m.rows(),
                    new_col_index_division % m.cols()
                );
                prev_row_index_division = new_row_index_division;
                prev_col_index_division = new_col_index_division;
                prev_matrix_rows_dim = m.rows();
                prev_matrix_cols_dim = m.cols();
            }
            /**Then i multiply for the first matrix element*/
            value *= (**matrices.begin())(
                static_cast<long long>(floor(prev_row_index_division / prev_matrix_rows_dim)),
                static_cast<long long>(floor(prev_col_index_division / prev_matrix_cols_dim))
                );
            (*result)(i, j) = value;
        }
    }
    return result;
}


/**BE CAREFUL: nth starts from 0*/
MatrixXd* tools::sigmaX(int nth, const int dim) {
    auto x = MatrixXd(2, 2); 
    x.setZero();
    x(0, 1) = 1;
    x(1, 0) = 1;

    int dim_1 = static_cast<int>(pow(2, nth));
    int dim_2 = static_cast<int>(pow(2, dim - nth - 1));

    auto x_nth = tools::kronecker_product({
        &MatrixXd(dim_1, dim_1).setIdentity(),
        &x,
        &MatrixXd(dim_2, dim_2).setIdentity()
        });

    return x_nth;
}
/**BE CAREFUL: nth starts from 0*/
MatrixXd* tools::sigmaY(int nth, const int dim) {
    auto y = MatrixXd(2, 2);
    y.setZero();
    y(0, 1) = -1;
    y(1, 0) = 1;

    int dim_1 = static_cast<int>(pow(2, nth));
    int dim_2 = static_cast<int>(pow(2, dim - nth - 1));

    auto y_nth = tools::kronecker_product({
        &MatrixXd(dim_1, dim_1).setIdentity(),
        &y,
        &MatrixXd(dim_2, dim_2).setIdentity()
        });

    return y_nth;
}
/**BE CAREFUL: nth starts from 0*/
MatrixXd* tools::sigmaZ(int nth, const int dim) {
    auto z = MatrixXd(2, 2);
    z.setZero();
    z(0, 0) = 1;
    z(1, 1) = -1;

    int dim_1 = static_cast<int>(pow(2, nth));
    int dim_2 = static_cast<int>(pow(2, dim - nth - 1));

    auto z_nth = tools::kronecker_product({
        &MatrixXd(dim_1, dim_1).setIdentity(),
        &z,
        &MatrixXd(dim_2, dim_2).setIdentity()
        });

    return z_nth;
}
void tools::generate_mb_sigma_operators(const int dim, const std::string& path) {

    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }
    else { //Deletes all the files inside to avoid clutter and confusion
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            std::filesystem::remove_all(entry.path());
        }
    }

    std::ofstream output;
    std::ostringstream full_path;
    cout << "x ";
    for (int i = 0; i < dim; ++i) {
        cout << i;
        auto sigmaX_i = sigmaX(i, dim);
        int size = sigmaX_i->rows();
        full_path << path << Pauli::X << "_sigma_" << i << ".bin";
        output.open(full_path.str(), std::ios::binary);

        /**Saves the size of the matrix. (Since it's a square matrix it's needed just one dimension)*/
        output.write(
            reinterpret_cast<const char*>(&size),
            sizeof(size)
        );
        /**Saves the whole matrix as a unique array right after the dimension bytes*/
        output.write(
            reinterpret_cast<const char*>(sigmaX_i->data()),
            sizeof(sigmaX_i->data()[0]) * size * size //Uses the size of one element as a reference since it's a homogeneous matrix
        );
        output.close();
        delete sigmaX_i;
        full_path.str("");
        full_path.clear();
    }
    cout << endl;
    cout << "y ";
    for (int i = 0; i < dim; ++i) {
        cout << i;
        auto sigmaY_i = sigmaY(i, dim);
        int size = sigmaY_i->rows();
        full_path << path << Pauli::Y << "_sigma_" << i << ".bin";
        output.open(full_path.str(), std::ios::binary);
        output.write(
            reinterpret_cast<const char*>(&size),
            sizeof(size)
        );

        output.write(
            reinterpret_cast<const char*>(sigmaY_i->data()),
            sizeof(sigmaY_i->data()[0]) * size * size
        );
        output.close();
        delete sigmaY_i;
        full_path.str("");
        full_path.clear();
    }
    cout << endl;
    cout << "z ";
    for (int i = 0; i < dim; ++i) {
        cout << i;
        auto sigmaZ_i = sigmaZ(i, dim);
        int size = sigmaZ_i->rows();
        full_path << path << Pauli::Z << "_sigma_" << i << ".bin";
        output.open(full_path.str(), std::ios::binary);
        output.write(
            reinterpret_cast<const char*>(&size),
            sizeof(size)
        );
        output.write(
            reinterpret_cast<const char*>(sigmaZ_i->data()),
            sizeof(sigmaZ_i->data()[0]) * size * size
        );
        output.close();
        delete sigmaZ_i;
        full_path.str("");
        full_path.clear();
    }
    cout << endl;
}
MatrixXd* tools::load_sigma(int nth_qubit, const Pauli which, const std::string& path) {
    std::ostringstream full_path;
    
    /*std::string which_uppercase = which;
    for (auto& c : which_uppercase) {
        c = toupper(c);
    }*/
    full_path << path << which << "_sigma_" << nth_qubit << ".bin";
    std::ifstream input(full_path.str(), std::ios::binary);
    int size;
    input.read(
        reinterpret_cast<char*>(&size),
        sizeof(size)
    );
    auto sigma = new MatrixXd(size, size);
    input.read(
        reinterpret_cast<char*>(sigma->data()),
        sizeof(double) * size * size //rows * columns number of elements
    );
    input.close();
    return sigma;
}
/**0x 1x 2x ... 0y 1y 2y ...*/
std::vector<MatrixXd*> tools::load_all_sigma(const int dim, const std::string& path) {
    std::vector<MatrixXd*> all (dim*3);

    for (int i = 0; i < dim; ++i) {
        all[i] = load_sigma(i, Pauli::X);
    }
    for (int i = dim; i < 2*dim; ++i) {
        all[i] = load_sigma(i, Pauli::Y);
    }
    for (int i = 2*dim; i < 3*dim; ++i) {
        all[i] = load_sigma(i, Pauli::Z);
    }
    return all;
}

MatrixXd* tools::ising_hamiltonian(const double H, const double J, const int dim) {
    auto on_site_parameter = new VectorXd(dim);
    on_site_parameter->setRandom();

    auto ham_on_site = new MatrixXd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
    ham_on_site->setZero();

    for (int i = 0; i < dim; ++i) {
        auto sz_i = sigmaZ(i, dim);
        *ham_on_site += 0.5 * H * (*on_site_parameter)[i] * (*sz_i);
        delete sz_i;
    }

    auto interaction_coupling = new MatrixXd(dim, dim);
    interaction_coupling->setRandom();

    auto ham_interaction = new MatrixXd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
    ham_interaction->setZero();

    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            auto sx_i = sigmaX(i, dim);
            auto sx_j = sigmaX(j, dim);
            *ham_interaction += 0.5 * J * (*interaction_coupling)(i, j) * (*sx_i) * (*sx_j);
            delete sx_i;
            delete sx_j;
        }
    }

    *ham_on_site += *ham_interaction;

    delete ham_interaction;

    return ham_on_site;
}
MatrixXcd* tools::reset(double signal_k, MatrixXcd* rho) {
    Eigen::Vector2d sig_state(2);
    sig_state << sqrt(1 - signal_k), sqrt(signal_k);

    MatrixXcd rho1 = sig_state * sig_state.transpose();

    auto traced = new MatrixXcd(rho->rows() / 2, rho->cols() / 2);

    long long rho_rows = rho->rows();
    long long rho_cols = rho->cols();

    /**Partial trace of rho*/
    for (int i = 0; i < rho_rows / 2; ++i) {
        for (int j = 0; j < rho_cols / 2; ++j) {
            (*traced)(i, j) = (*rho)(i, j) + (*rho)(i + rho_rows / 2, j + rho_cols / 2);
        }
    }

    auto result = tools::kronecker_product({ &rho1, traced });

    delete traced;

    return result;
}
MatrixXcd* tools::time_evolution_operator(const double dt, Eigen::DiagonalMatrix<double, Eigen::Dynamic>* D, MatrixXd* U, MatrixXd* U_inv) {
    auto op = new MatrixXcd(U->rows(), U->cols());
    auto phase_factors = new Eigen::DiagonalMatrix<complex<double>, Eigen::Dynamic>(D->rows());

    for (int i = 0; i < D->diagonal().size(); ++i) {
        /** exp(-i*D*dt) */
        phase_factors->diagonal()(i) = exp(-complex<double>(0, 1) * D->diagonal()(i) * dt);
    }

    *op = *U * *phase_factors * *U_inv;

    delete phase_factors;

    return op;
}

void tools::average_single(MatrixXcd* rho, std::vector<MatrixXd*> all, MatrixXd& output, int sample) {
    /*Row -> qubit, Cols -> average of j-th operator**/
    int dim = static_cast<int>(round(std::log2(rho->rows())));
    //X
    for (int i = 0; i < dim; ++i) {
        output(sample, i) = (*rho * *all[i]).trace().real();
    }
    //Y
    for (int i = dim; i < 2*dim; ++i) {
        output(sample, i) = -(*rho * *all[i]).trace().imag(); // -Im(...) since we are using real sigmaY.
    }
    //Z
    for (int i = 2*dim; i < 3*dim; ++i) {
        output(sample, i) = (*rho * *all[i]).trace().real();
    }
}
void tools::average_double(MatrixXcd* rho, std::vector<MatrixXd*> all, MatrixXd& output, int sample) {
    int dim = static_cast<int>(round(std::log2(rho->rows())));
    
    int counter = 3*dim;
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            //nb: counter++ first pass the value then increment.
            output(sample, counter++) = (*rho * *all[i] * *all[j]).trace().real();
            output(sample, counter++) = (-1 * *rho * *all[i + dim] * *all[j + dim]).trace().real();// x-1 since we are using the real sigmaY
            output(sample, counter++) = (*rho * *all[i + 2*dim] * *all[j + 2*dim]).trace().real();
        }
    }
}

void tools::clear_data_folder() {

    if (!std::filesystem::exists(data_output_path)) {
        std::filesystem::create_directories(data_output_path);
    }
    else { //Deletes all the files inside the directory
        for (const auto& entry : std::filesystem::directory_iterator(data_output_path)) {
            std::filesystem::remove_all(entry.path());
        }
    }

}
void tools::exportMatrixToCSV(MatrixXd* data, const std::string& filename) {
    std::ofstream file;
    

    if (!std::filesystem::exists(data_output_path)) {
        std::filesystem::create_directories(data_output_path);
    }

    file.open(data_output_path + filename + ".csv");
    for (int i = 0; i < data->rows(); ++i) {
        for (int j = 0; j < data->cols() - 1; ++j) {
            file << (*data)(i, j) << ",";
        }
        file << (*data)(i, data->cols() - 1);
        file << endl;
    }
    file.close();
}
void tools::exportVectorToCSV(VectorXd* data, const std::string& filename) {
    std::ofstream file;

    if (!std::filesystem::exists(data_output_path)) {
        std::filesystem::create_directories(data_output_path);
    }

    file.open(data_output_path + filename + ".csv");
    for (int i = 0; i < data->size() - 1; ++i) {
        file << (*data)[i];
        file << endl;
    }
    file << (*data)[data->size() - 1];
    file.close();
}
