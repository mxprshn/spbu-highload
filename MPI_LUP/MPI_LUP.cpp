#include <iostream>
#include <chrono>
#include <random>
#include <string>
#include <omp.h>
#include <mpi.h>

using namespace std;

uniform_real_distribution<double> unif(0, 1);
default_random_engine engine;

const int ROOT_PROC_ID = 0;
const int SEED = 73;

int process_id = -1;
bool is_root()
{
    if (process_id == -1)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    }
    
    return process_id == ROOT_PROC_ID;
}

double** copy_matrix(double** M, int n)
{
    auto copy = new double* [n];

    for (auto i = 0; i < n; i++)
    {
        copy[i] = new double[n];
        for (auto j = 0; j < n; j++)
        {
           copy[i][j] =M[i][j];
        }
    }

    return copy;
}

double** multiply(double** A, double** B, int n)
{
    auto mult = new double* [n];

    for (int i = 0; i < n; i++) 
    {
        mult[i] = new double[n];

        for (int j = 0; j < n; j++) 
        {
            mult[i][j] = 0;

            for (int k = 0; k < n; k++) 
            {
                mult[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return mult;
}

bool is_identity(double** M, int n)
{
    const double epsilon = 0.0000001;

    for (auto i = 0; i < n; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            if (i == j)
            {
                if (abs(M[i][j] - 1) > epsilon)
                {
                    return false;
                }                
            }
            else
            {
                if (abs(M[i][j]) > epsilon)
                {
                    return false;
                }
            }

        }
    }

    return true;
}

double** generate_random_matrix(int n)
{
    auto matrix = new double* [n];

    for (auto i = 0; i < n; i++)
    {
        matrix[i] = new double[n];
        for (auto j = 0; j < n; j++)
        {
            matrix[i][j] = unif(engine);
        }
    }

    return matrix;
}

double** generate_identity_matrix(int n)
{
    auto matrix = new double* [n];

    for (auto i = 0; i < n; i++)
    {
        matrix[i] = new double[n];

        for (auto j = 0; j < n; j++)
        {
            matrix[i][j] = i != j ? 0.0 : 1.0;
        }
    }

    return matrix;
}

double** generate_zero_matrix(int n)
{
    auto matrix = new double* [n];

    for (auto i = 0; i < n; i++)
    {
        matrix[i] = new double[n];

        for (auto j = 0; j < n; j++)
        {
            matrix[i][j] = 0.0;
        }
    }

    return matrix;
}

void print_matrix(double** M, int n)
{
    std::cout << "[" << endl;
    for (auto i = 0; i < n; ++i)
    {
        std::cout << "\t";

        for (auto j = 0; j < n; ++j)
        {
            std::cout << M[i][j] << " ";
        }

        std::cout << endl;
    }
    std::cout << "]" << endl;
}

double** lup_decomposition(double** A, int n)
{
    auto process_count = 0;
    auto process_id = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    auto P = generate_identity_matrix(n);
    auto pivot_time_sum = 0.0;

    for (auto i = 0; i < n; i++)
    {
        if (is_root())
        {
            auto row = i;
            auto max = 0.0;

            auto pivot_start = MPI_Wtime();

            for (auto r = i; r < n; r++)
            {
                auto current = A[r][i];

                auto current_abs = abs(current);
                if (current_abs > max)
                {
                    max = current_abs;
                    row = r;
                }
            }

            if (row != i)
            {
                for (auto q = 0; q < n; q++)
                {
                    auto tmp = P[i][q];
                    P[i][q] = P[row][q];
                    P[row][q] = tmp;
                    tmp = A[i][q];
                    A[i][q] = A[row][q];
                    A[row][q] = tmp;
                }
            }

            pivot_time_sum += (MPI_Wtime() - pivot_start);

            for (auto j = 1; j < process_count; j++)
            {
                MPI_Send(&A[i][i], n - i, MPI_DOUBLE, j, 1, MPI_COMM_WORLD);
            }

            for (auto j = i + 1; j < n; j++)
            {
                MPI_Send(&A[j][i], n - i, MPI_DOUBLE, j % (process_count - 1) + 1, j + 1, MPI_COMM_WORLD);
            }

            for (auto j = 1; j < process_count; j++)
            {
                MPI_Send(nullptr, 0, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
            }

            MPI_Status status;

            for (auto j = i + 1; j < n; j++)
            {                
                MPI_Recv(&A[j][i], n - i, MPI_DOUBLE, j % (process_count - 1) + 1, j + 1, MPI_COMM_WORLD, &status);
            }
        }
        else
        {
            auto primary_row = new double[n - i];
            auto row = new double[n - i];

            MPI_Status status;
            MPI_Recv(primary_row, n - i, MPI_DOUBLE, ROOT_PROC_ID, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG != 0)
            {
                while (true)
                {
                    MPI_Recv(row, n - i, MPI_DOUBLE, ROOT_PROC_ID, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                    if (status.MPI_TAG == 0)
                    {
                        break;
                    }

                    row[0] /= primary_row[0];

                    for (auto k = 1; k < n - i; k++)
                    {
                        row[k] -= row[0] * primary_row[k];
                    }

                    MPI_Send(row, n - i, MPI_DOUBLE, ROOT_PROC_ID, status.MPI_TAG, MPI_COMM_WORLD);
                }
            }
        }
    }

    if (is_root())
    {
        std::cout << "Elapsed (pivoting): " << pivot_time_sum << " s" << endl;
    }
    
    return P;
}

void exchange(double** M, int n)
{
    for (auto i = 0; i < n; i++)
    {
        MPI_Bcast(M[i], n, MPI_DOUBLE, ROOT_PROC_ID, MPI_COMM_WORLD);
    }
}

double** get_inverse(double** LU, double** P, int n) {

    auto process_count = 0;
    auto process_id = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    auto I_column = new double[n + 1] { 0.0 };
    auto I = process_id == ROOT_PROC_ID ? generate_zero_matrix(n) : nullptr;
    auto Y = new double[n + 1] { 0.0 };

    if (process_id != ROOT_PROC_ID)
    {
        for (auto q = 0; q < n; q++)
        {
            if (q % (process_count - 1) + 1 == process_id)
            {
                for (auto i = 0; i < n; i++)
                {
                    Y[i] = P[i][q];
                    auto j = 0;
                    while (j < i) {
                        Y[i] += -LU[i][j] * Y[j];
                        j++;
                    }
                }

                for (auto i = n - 1; i > -1; i--)
                {
                    I_column[i] = Y[i];
                    auto j = i + 1;
                    while (j < n)
                    {
                        I_column[i] += -LU[i][j] * I_column[j];
                        j++;
                    }
                    I_column[i] = I_column[i] / LU[i][i];
                }

                I_column[n] = q;

                MPI_Send(I_column, n + 1, MPI_DOUBLE, ROOT_PROC_ID, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        auto chunk_count = n / (process_count - 1);
        auto columns_received = 0;

        while (columns_received != n)
        {
            for (auto i = 1; i < process_count; i++)
            {
                MPI_Status status;
                MPI_Recv(I_column, n + 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                for (auto j = 0; j < n; j++)
                {
                    I[j][int(I_column[n])] = I_column[j];
                }

                columns_received++;

                if (columns_received == n)
                {
                    break;
                }
            }
        }
    }

    return I;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    auto process_count = 0;
    auto process_id = 0;

    auto n = stoi(argv[1]);

    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    if (is_root())
    {
        cout << "Processes: " << process_count << endl;
        cout << "n: " << n << endl;
    }

    engine.seed(SEED);

    auto A = is_root() ? generate_random_matrix(n) : generate_zero_matrix(n);
    auto A_copy = is_root() ? copy_matrix(A, n) : nullptr;

    auto start_time = is_root() ? MPI_Wtime() : 0;

    auto P = lup_decomposition(A, n);

    auto exc_start = is_root() ? MPI_Wtime() : 0;

    exchange(A, n);
    exchange(P, n);

    auto exc_time = is_root() ? MPI_Wtime() - exc_start : 0;

    auto inv_start = is_root() ? MPI_Wtime() : 0;
    auto I = get_inverse(A, P, n);

    auto time = is_root() ? MPI_Wtime() - start_time : 0;
    auto inv_time = is_root() ? MPI_Wtime() - inv_start : 0;

    if (is_root())
    {
        std::cout << "Elapsed: " << time << " s" << endl;
        std::cout << "Elapsed (inverse): " << inv_time << " s" << endl;
        std::cout << "Elapsed (exchange): " << exc_time << " s" << endl;
        auto mult = multiply(A_copy, I, n);
        if (is_identity(mult, n))
        {
            std::cout << "Identity check passed!" << endl;
        }
        else
        {
            std::cout << "Identity check not passed!" << endl;
        }
    }

    MPI_Finalize();

    return 0;
}
