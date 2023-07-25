#include "tools.h";

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::MatrixX;
using Eigen::SelfAdjointEigenSolver;

using std::complex;
using std::floor;
using std::cout;
using std::endl;

std::string tools::sigma_operators_path;
std::string tools::data_output_path;
int tools::dim;

void tools::set_system_dimension(int dim) {
    tools::dim = dim;
}
void tools::set_sigma_operators_path(const std::string& new_path) {
    tools::sigma_operators_path = new_path;
}
void tools::set_data_output_path(const std::string& new_path) {
    tools::data_output_path = new_path;
}

template<typename T>
MatrixX<T>* tools::kronecker_product(const std::vector<MatrixX<T>>& matrices);
template<typename T>
MatrixX<T>* tools::kronecker_product(const std::initializer_list<MatrixX<T>*> matrices) {

    long long int rows_dim = 1;
    long long int cols_dim = 1;

    for (int i = 0; i < matrices.size(); ++i) {
        rows_dim *= (*(matrices.begin() + i))->rows();
        cols_dim *= (*(matrices.begin() + i))->cols();
    }

    MatrixX<T>* result = new MatrixX<T>(rows_dim, cols_dim);

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
                const MatrixX<T>& m = **(matrices.end() - 1 - l);

                long long new_row_index_division = static_cast<long long>(floor(prev_row_index_division / prev_matrix_rows_dim));
                long long new_col_index_division = static_cast<long long>(floor(prev_col_index_division / prev_matrix_cols_dim));
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
MatrixXd* tools::sigmaX(int nth) {
    MatrixXd x = MatrixXd(2, 2); 
    x.setZero();
    x(0, 1) = 1;
    x(1, 0) = 1;

    int dim_1 = static_cast<int>(pow(2, nth));
    int dim_2 = static_cast<int>(pow(2, dim - nth - 1));

    MatrixXd* x_nth = tools::kronecker_product({
        &MatrixXd(dim_1, dim_1).setIdentity(),
        &x,
        &MatrixXd(dim_2, dim_2).setIdentity()
        });

    return x_nth;
}
/**BE CAREFUL: nth starts from 0*/
MatrixXd* tools::sigmaY(int nth) {
    MatrixXd y = MatrixXd(2, 2);
    y.setZero();
    y(0, 1) = -1;
    y(1, 0) = 1;

    int dim_1 = static_cast<int>(pow(2, nth));
    int dim_2 = static_cast<int>(pow(2, dim - nth - 1));

    MatrixXd* y_nth = tools::kronecker_product({
        &MatrixXd(dim_1, dim_1).setIdentity(),
        &y,
        &MatrixXd(dim_2, dim_2).setIdentity()
        });

    return y_nth;
}
/**BE CAREFUL: nth starts from 0*/
MatrixXd* tools::sigmaZ(int nth) {
    MatrixXd z = MatrixXd(2, 2);
    z.setZero();
    z(0, 0) = 1;
    z(1, 1) = -1;

    int dim_1 = static_cast<int>(pow(2, nth));
    int dim_2 = static_cast<int>(pow(2, dim - nth - 1));

    MatrixXd* z_nth = tools::kronecker_product({
        &MatrixXd(dim_1, dim_1).setIdentity(),
        &z,
        &MatrixXd(dim_2, dim_2).setIdentity()
        });

    return z_nth;
}

SparseMatrix<double>* tools::sparseSigmaX(int nth) {
    int matrix_dim = (int)pow(2, dim);
    int n_blocks = (int)pow(2, nth + 1);
    int shift = matrix_dim / n_blocks;
    SparseMatrix<double>* x_nth = new SparseMatrix<double>(matrix_dim, matrix_dim);
    x_nth->reserve(1);

    int row = shift;
    for (int i = 0; i < n_blocks; ++i) {
        for (int j = i * shift; j < shift * (i + 1); ++j) {
            x_nth->insert(row, j) = 1;
            ++row;
        }
        if (i % 2 == 0) {
            row -= 2 * shift;
        }
        else {
            row+= 2 * shift;
        }
    }
    x_nth->makeCompressed();

    return x_nth;
}
SparseMatrix<double>* tools::sparseSigmaY(int nth) {
    int matrix_dim = (int)pow(2, dim);
    int n_blocks = (int)pow(2, nth + 1);
    int shift = matrix_dim / n_blocks;
    SparseMatrix<double>* y_nth = new SparseMatrix<double>(matrix_dim, matrix_dim);
    y_nth->reserve(1);

    int row = shift;
    for (int i = 0; i < n_blocks; ++i) {
        for (int j = i * shift; j < shift * (i + 1); ++j) {
            y_nth->insert(row, j) = 1 - (i % 2)*2;
            ++row;
        }
        if (i % 2 == 0) {
            row -= 2 * shift;
        }
        else {
            row += 2 * shift;
        }
    }
    y_nth->makeCompressed();

    return y_nth;
}
SparseMatrix<double>* tools::sparseSigmaZ(int nth) {
    int matrix_dim = (int)pow(2, dim);
    int n_blocks = (int)pow(2, nth + 1);
    int shift = matrix_dim / n_blocks;
    SparseMatrix<double>* z_nth = new SparseMatrix<double>(matrix_dim, matrix_dim);
    z_nth->reserve(1);

    for (int i = 0; i < n_blocks; ++i) {
        for (int j = i * shift; j < shift * (i + 1); ++j) {
            z_nth->insert(j, j) = 1 - (i % 2) * 2;
        }
    }
    z_nth->makeCompressed();

    return z_nth;
}

void tools::generate_mb_sigma_operators(const std::string& path) {

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
        auto sigmaX_i = sigmaX(i);
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
        auto sigmaY_i = sigmaY(i);
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
        auto sigmaZ_i = sigmaZ(i);
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

    full_path << path << which << "_sigma_" << nth_qubit << ".bin";
    std::ifstream input(full_path.str(), std::ios::binary);
    int size;
    input.read(
        reinterpret_cast<char*>(&size),
        sizeof(size)
    );
    MatrixXd* sigma = new MatrixXd(size, size);
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
        all[i + dim] = load_sigma(i, Pauli::Y);
        all[i + 2*dim] = load_sigma(i, Pauli::Z);
    }
    return all;
}

std::vector<SparseMatrix<double>*> tools::generate_all_sigma() {
    std::vector<SparseMatrix<double>*> sigma(dim * 3);
    for (int i = 0; i < dim; ++i) {
        sigma[i] = sparseSigmaX(i);
        sigma[i + dim] = sparseSigmaY(i);
        sigma[i + 2*dim] = sparseSigmaZ(i);
    }
    return sigma;
}

MatrixXd* tools::ising_hamiltonian(const VectorXd* h, const double J) {

    MatrixXd* ham_on_site = new MatrixXd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
    ham_on_site->setZero();

    for (int i = 0; i < dim; ++i) {
        SparseMatrix<double>* sz_i = sparseSigmaZ(i);
        *ham_on_site += 0.5 * (*h)[i] * (*sz_i);
        delete sz_i;
    }

    MatrixXd* interaction_coupling = new MatrixXd(dim, dim);
    interaction_coupling->setRandom();//default [-1, 1]

    MatrixXd* ham_interaction = new MatrixXd(static_cast<long long>(pow(2, dim)), static_cast<long long>(pow(2, dim)));
    ham_interaction->setZero();

    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            SparseMatrix<double>* sx_i = sparseSigmaX(i);
            SparseMatrix<double>* sx_j = sparseSigmaX(j);
            *ham_interaction += 0.5 * J * (*interaction_coupling)(i, j) * (*sx_i) * (*sx_j);
            delete sx_i;
            delete sx_j;
        }
    }

    *ham_on_site += *ham_interaction;

    delete ham_interaction;

    return ham_on_site;
}

void tools::wash_out(MatrixXcd** rho, int input_size, MatrixXcd* time_ev_op_s, MatrixXcd* time_ev_op_d) {
    VectorXd random_input(input_size);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> distr(0, 1);

    for (int i = 0; i < random_input.size(); ++i) {
        random_input[i] = distr(gen);
    }

    for (int i = 0; i < input_size; ++i) {
        reset(rho, random_input[i]);
        **rho = *time_ev_op_s * **rho * *time_ev_op_d;
    }
}
void tools::reset(MatrixXcd** rho, double signal_k) {
    Eigen::Vector2d sig_state(2);
    sig_state << sqrt(1 - signal_k), sqrt(signal_k);

    MatrixXcd rho1 = sig_state * sig_state.transpose();

    long long rho_rows = (*rho)->rows();
    long long rho_cols = (*rho)->cols();

    MatrixXcd* traced = new MatrixXcd(rho_rows / 2, rho_cols / 2);


    /**Partial trace of rho*/
    for (int i = 0; i < rho_rows / 2; ++i) {
        for (int j = 0; j < rho_cols / 2; ++j) {
            (*traced)(i, j) = (**rho)(i, j) + (**rho)(i + rho_rows / 2, j + rho_cols / 2);
        }
    }

    MatrixXcd* result = tools::kronecker_product({ &rho1, traced });

    delete traced;
    delete* rho;

    *rho = result;
}
MatrixXcd* tools::time_evolution_operator(MatrixXd* hamiltonian, const double dt) {
    SelfAdjointEigenSolver<MatrixXd> es(*hamiltonian);

    MatrixXd* U = new MatrixXd(hamiltonian->rows(), hamiltonian->cols());
    (*U) << es.eigenvectors();

    Eigen::DiagonalMatrix<double, Eigen::Dynamic>* D = new Eigen::DiagonalMatrix<double, Eigen::Dynamic>(hamiltonian->rows());
    D->diagonal() << es.eigenvalues();

    MatrixXd* U_inv = new MatrixXd(hamiltonian->rows(), hamiltonian->cols());
    *U_inv << U->transpose();

    MatrixXcd* exp_s = new MatrixXcd(U->rows(), U->cols());
    Eigen::DiagonalMatrix<complex<double>, Eigen::Dynamic>* phase_factors = new Eigen::DiagonalMatrix<complex<double>, Eigen::Dynamic>(D->rows());

    for (int i = 0; i < D->diagonal().size(); ++i) {
        /** exp(-i*D*dt) */
        phase_factors->diagonal()(i) = exp(-complex<double>(0, 1) * D->diagonal()(i) * dt);
    }

    *exp_s = *U * *phase_factors * *U_inv;

    delete phase_factors;
    delete U;
    delete D;
    delete U_inv;

    return exp_s;
}
void tools::average_single(MatrixXcd* rho, std::vector<SparseMatrix<double>*> sigma, MatrixXd* output, int sample) {
    /*Row -> qubit, Cols -> average of j-th operator**/
    //X
    for (int i = 0; i < dim; ++i) {
        (*output)(sample, i) = (*rho * *sigma[i]).trace().real();
    }
    //Y
    for (int i = dim; i < 2*dim; ++i) {
        (*output)(sample, i) = -(*rho * *sigma[i]).trace().imag(); // -Im(...) since we are using real sigmaY.
    }
    
    //Z
    for (int i = 2*dim; i < 3*dim; ++i) {
        (*output)(sample, i) = (*rho * *sigma[i]).trace().real();
    }
}
void tools::average_double(MatrixXcd* rho, std::vector<SparseMatrix<double>*> sigma, MatrixXd* output, int sample) {
    int counter = 3*dim;
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            //nb: counter++ first pass the value then increment.
            (*output)(sample, counter++) = (*rho * *sigma[i] * *sigma[j]).trace().real();
            (*output)(sample, counter++) = (-1 * *rho * *sigma[i + dim] * *sigma[j + dim]).trace().real();// x-1 since we are using the real sigmaY
            (*output)(sample, counter++) = (*rho * *sigma[i + 2 * dim] * *sigma[j + 2 * dim]).trace().real();
        }
    }
}
MatrixXd* tools::measure_output(MatrixXcd** rho, std::vector<SparseMatrix<double>*> sigma, VectorXd* signal, MatrixXcd* time_ev_op_s, MatrixXcd* time_ev_op_d, int tau) {
    /**We are doing 3*dim + 3*(dim-1)*dim/2 measurements per sample i.e. we have 3*dim + 3*(dim-1)*dim/2 columns and M rows*/
    int n_meas = 3 * dim + 3 * dim * (dim - 1) / 2;
    MatrixXd* measurements = new MatrixXd(signal->size() - tau, n_meas);

    /**Here starts the routine*/
    for (int s = 0; s < signal->size(); ++s) {
        reset(rho, (*signal)[s]);
        **rho = *time_ev_op_s * **rho * *time_ev_op_d;
        if (s >= tau) {
            average_single(*rho, sigma, measurements, s - tau);
            average_double(*rho, sigma, measurements, s - tau);
        }
    }

    return measurements;
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
