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
using std::complex;
using std::floor;
using std::cout;
using std::endl;

MatrixXcd* tools::kronecker_product(const std::vector<MatrixXcd>& matrices) {

    long long int rows_dim = 1;
    long long int cols_dim = 1;


    for (int i = 0; i < matrices.size(); ++i) {
        rows_dim *= (*(matrices.begin() + i)).rows();
        cols_dim *= (*(matrices.begin() + i)).cols();
    }

    auto result = new MatrixXcd(rows_dim, cols_dim);

#pragma omp parallel for default(none) shared(cout, rows_dim, cols_dim, matrices, result)
    for (int i = 0; i < rows_dim; ++i) {
        for (int j = 0; j < cols_dim; ++j) {
            /**It starts by calculating the first factor from the last matrix*/
            complex<double> value = (*(matrices.end() - 1))(i % (*(matrices.end() - 1)).rows(), j % (*(matrices.end() - 1)).cols());

            /**We start from i and j*/
            long long prev_row_index_division = i;
            long long prev_col_index_division = j;

            long long prev_matrix_rows_dim = (*(matrices.end() - 1)).rows();
            long long prev_matrix_cols_dim = (*(matrices.end() - 1)).cols();

            /**We go from the N - 1 matrix to the 2 matrix*/
            for (int l = 1; l < matrices.size() - 1; ++l) {
                const MatrixXcd& m = *(matrices.end() - 1 - l);

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
            value *= (*matrices.begin())(
                static_cast<long long>(floor(prev_row_index_division / prev_matrix_rows_dim)),
                static_cast<long long>(floor(prev_col_index_division / prev_matrix_cols_dim))
                );
            (*result)(i, j) = value;
        }
    }
    return result;
}
MatrixXcd* tools::kronecker_product(const std::initializer_list<MatrixXcd*> matrices) {

    long long int rows_dim = 1;
    long long int cols_dim = 1;

    for (int i = 0; i < matrices.size(); ++i) {
        rows_dim *= (*(matrices.begin() + i))->rows();
        cols_dim *= (*(matrices.begin() + i))->cols();
    }

    auto result = new MatrixXcd(rows_dim, cols_dim);

#pragma omp parallel for default(none) shared(cout, rows_dim, cols_dim, matrices, result)
    for (int i = 0; i < rows_dim; ++i) {

        for (int j = 0; j < cols_dim; ++j) {

            /**It starts by calculating the first factor from the last matrix*/
            complex<double> value = (**(matrices.end() - 1))(i % (**(matrices.end() - 1)).rows(), j % (**(matrices.end() - 1)).cols());

            /**We start from i and j*/
            long long prev_row_index_division = i;
            long long prev_col_index_division = j;

            long long prev_matrix_rows_dim = (*(matrices.end() - 1))->rows();
            long long prev_matrix_cols_dim = (*(matrices.end() - 1))->cols();

            /**We go from the N - 1 matrix to the 2 matrix*/
            for (int l = 1; l < matrices.size() - 1; ++l) {
                const MatrixXcd& m = **(matrices.end() - 1 - l);

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
MatrixXcd* tools::sigmaX(int nth, const int dim) {
    auto x = MatrixXcd(2, 2);
    x.setZero();
    x(0, 1) = 1;
    x(1, 0) = 1;

    int dim_1 = static_cast<int>(pow(2, nth));
    int dim_2 = static_cast<int>(pow(2, dim - nth - 1));

    auto x_nth = tools::kronecker_product({
        &MatrixXcd(dim_1, dim_1).setIdentity(),
        &x,
        &MatrixXcd(dim_2, dim_2).setIdentity()
        });

    return x_nth;
}
/**BE CAREFUL: nth starts from 0*/
MatrixXcd* tools::sigmaY(int nth, const int dim) {
    auto y = MatrixXcd(2, 2);
    y.setZero();
    y(0, 1) = complex<double>(0, -1);
    y(1, 0) = complex<double>(0, 1);

    int dim_1 = static_cast<int>(pow(2, nth));
    int dim_2 = static_cast<int>(pow(2, dim - nth - 1));

    auto y_nth = tools::kronecker_product({
        &MatrixXcd(dim_1, dim_1).setIdentity(),
        &y,
        &MatrixXcd(dim_2, dim_2).setIdentity()
        });

    return y_nth;
}
/**BE CAREFUL: nth starts from 0*/
MatrixXcd* tools::sigmaZ(int nth, const int dim) {
    auto z = MatrixXcd(2, 2);
    z.setZero();
    z(0, 0) = 1;
    z(1, 1) = -1;

    int dim_1 = static_cast<int>(pow(2, nth));
    int dim_2 = static_cast<int>(pow(2, dim - nth - 1));

    auto z_nth = tools::kronecker_product({
        &MatrixXcd(dim_1, dim_1).setIdentity(),
        &z,
        &MatrixXcd(dim_2, dim_2).setIdentity()
        });

    return z_nth;
}

MatrixXcd* tools::ising_hamiltonian(const double H, const double J, const int dim) {
    auto on_site_parameter = new VectorXd(dim);
    on_site_parameter->setRandom();

    auto ham_on_site = new MatrixXcd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
    ham_on_site->setZero();

    for (int i = 0; i < dim; ++i) {
        auto sz_i = sigmaZ(i, dim);
        *ham_on_site += 0.5 * H * (*on_site_parameter)[i] * (*sz_i);
        delete sz_i;
    }

    auto interaction_coupling = new MatrixXd(dim, dim);
    interaction_coupling->setRandom();

    auto ham_interaction = new MatrixXcd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
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
    Eigen::Vector2cd sig_state(2);
    sig_state << sqrt(1 - signal_k), sqrt(signal_k);

    MatrixXcd rho1 = sig_state * sig_state.transpose().conjugate();

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

MatrixXcd* tools::time_evolution_operator(const double dt, Eigen::DiagonalMatrix<complex<double>, Eigen::Dynamic>* D, MatrixXcd* U, MatrixXcd* U_inv) {
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

void tools::average_single(MatrixXcd* rho, std::vector<MatrixXcd*> all, MatrixXd& output, int sample) {
    /*Row -> qubit, Cols -> average of j-th operator**/
    int dim = static_cast<int>(round(std::log2(rho->rows())));

    /*for (int i = 0; i < dim; ++i) {
        auto sx = load_sigma(i, "x");
        auto sy = load_sigma(i, "y");
        auto sz = load_sigma(i, "z");
  
        (*meas)(i, 0) = (*rho * *sx).trace().real();
        (*meas)(i, 1) = (*rho * *sy).trace().real();
        (*meas)(i, 2) = (*rho * *sz).trace().real();
        delete sx;
        delete sy;
        delete sz;
    }*/
    for (int i = 0; i < 3*dim; ++i) {
        output(sample, i) = (*rho * *all[i]).trace().real();
    }
}

//MatrixXd* tools::average_double(MatrixXcd* rho){
//    int dim = static_cast<int>(round(std::log2(rho->rows())));
//    auto meas = new MatrixXd(dim, 9*dim);
//
//    for (int i = 0; i < dim; ++i) {
//        auto sx_i = sigmaX(i, dim);
//        auto sy_i = sigmaY(i, dim);
//        auto sz_i = sigmaZ(i, dim);
//        for (int j = 0; j < dim; ++j) {
//            auto sx_j = sigmaX(j, dim);
//            auto sy_j = sigmaY(j, dim);
//            auto sz_j = sigmaZ(j, dim);
//
//            (*meas)(i, 9*j    ) = (*sx_i * *sx_j * *rho).trace().real();
//            (*meas)(i, 9*j + 1) = (*sx_i * *sy_j * *rho).trace().real();
//            (*meas)(i, 9*j + 2) = (*sx_i * *sz_j * *rho).trace().real();
//
//            (*meas)(i, 9*j + 3) = (*sy_i * *sx_j * *rho).trace().real();
//            (*meas)(i, 9*j + 4) = (*sy_i * *sy_j * *rho).trace().real();
//            (*meas)(i, 9*j + 5) = (*sy_i * *sz_j * *rho).trace().real();
//
//            (*meas)(i, 9*j + 6) = (*sz_i * *sx_j * *rho).trace().real();
//            (*meas)(i, 9*j + 7) = (*sz_i * *sy_j * *rho).trace().real();
//            (*meas)(i, 9*j + 8) = (*sz_i * *sz_j * *rho).trace().real();
//
//            delete sx_j;
//            delete sy_j;
//            delete sz_j;
//        }
//        delete sx_i;
//        delete sy_i;
//        delete sz_i;
//    }
//    return meas;
//}

void tools::average_double(MatrixXcd* rho, std::vector<MatrixXcd*> all, MatrixXd& output, int sample) {
    int dim = static_cast<int>(round(std::log2(rho->rows())));
    //auto meas = new VectorXd(3 * dim * (dim - 1) / 2);

    //Loading the operators each time
    //int counter = 0;
    //for (int i = 0; i < dim; ++i) {
    //    auto sx_i = load_sigma(i, "x");
    //    auto sy_i = load_sigma(i, "y");
    //    auto sz_i = load_sigma(i, "z");
    //    for (int j = i + 1; j < dim; ++j) {

    //        auto sx_j = load_sigma(j, "x");
    //        auto sy_j = load_sigma(j, "y");
    //        auto sz_j = load_sigma(j, "z");

    //        (*meas)(counter) = (*sx_i * *sx_j * *rho).trace().real();
    //        ++counter;
    //        (*meas)(counter) = (*sy_i * *sy_j * *rho).trace().real();
    //        ++counter;
    //        (*meas)(counter) = (*sz_i * *sz_j * *rho).trace().real();
    //        ++counter;

    //        delete sx_j;
    //        delete sy_j;
    //        delete sz_j;
    //    }
    //    delete sx_i;
    //    delete sy_i;
    //    delete sz_i;
    //}

    //Loading the operators once and referencing them in memory each time
    //int counter = 0;
    //for (int i = 0; i < dim; ++i) {
    //    for (int j = i + 1; j < dim; ++j) {
    //        (*meas)(counter) = (*all[3*i] * *all[3*j] * *rho).trace().real();
    //        ++counter;
    //        (*meas)(counter) = (*all[3*i + 1] * *all[3*j + 1] * *rho).trace().real();
    //        ++counter;
    //        (*meas)(counter) = (*all[3*i + 2] * *all[3*j + 2] * *rho).trace().real();
    //        ++counter;
    //    }
    //}
    
    int counter = 3*dim;
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            //nb: counter++ first pass the value then increment.
            output(sample, counter++) = (*rho * *all[3 * i] * *all[3 * j]).trace().real();
            output(sample, counter++) = (*rho * *all[3 * i + 1] * *all[3 * j + 1]).trace().real();
            output(sample, counter++) = (*rho * *all[3 * i + 2] * *all[3 * j + 2]).trace().real();
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

void tools::clear_data_folder() {
    const std::string path = "../../../data/";

    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }
    else { //Deletes all the files inside the directory
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            std::filesystem::remove_all(entry.path());
        }
    }

}

std::string tools::sigma_operators_path;
void tools::set_sigma_operators_path(const std::string& new_path) {
    tools::sigma_operators_path = new_path;
}

std::string tools::data_output_path;
void tools::set_data_output_path(const std::string& new_path) {
    tools::data_output_path = new_path;
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
    for (int i = 0; i < dim; ++i) {
        cout << i;
        auto sigmaX_i = sigmaX(i, dim);
        int size = sigmaX_i->rows();
        std::ostringstream full_path;
        full_path << path << "/sigmaX_" << i << ".bin";
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
        cout << " x";

        full_path << path << "/sigmaY_" << i << ".bin";
        auto sigmaY_i = sigmaY(i, dim);
        size = sigmaY_i->rows();
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
        cout << " y";

        full_path << path << "/sigmaZ_" << i << ".bin";
        auto sigmaZ_i = sigmaZ(i, dim);
        size = sigmaZ_i->rows();
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
        cout << " z" << endl;

    }
}

MatrixXcd* tools::load_sigma(int nth_qubit, const std::string& which, const std::string& path) {
    std::ostringstream full_path;
    
    std::string which_uppercase = which;
    for (auto& c : which_uppercase) {
        c = toupper(c);
    }
    full_path << path << "/sigma" << which_uppercase << "_" << nth_qubit << ".bin";
    std::ifstream input(full_path.str(), std::ios::binary);
    int size;
    input.read(
        reinterpret_cast<char*>(&size),
        sizeof(size)
    );
    auto sigma = new MatrixXcd(size, size);
    input.read(
        reinterpret_cast<char*>(sigma->data()),
        sizeof(std::complex<double>) * size * size //rows * columns number of elements
    );
    input.close();
    return sigma;
}

/**0x 0y 0z 1x 1y 1z ...*/
std::vector<MatrixXcd*> tools::load_all_sigma(const int dim, const std::string& path) {
    std::vector<MatrixXcd*> all (dim*3);
    for (int i = 0, j = 0; i < 3*dim; i+=3, ++j) {
        all[i] = load_sigma(j, "x");
        all[i + 1] = load_sigma(j, "y");
        all[i + 2] = load_sigma(j, "z");
    }
    return all;
}