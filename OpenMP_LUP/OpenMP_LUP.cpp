#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

uniform_real_distribution<double> unif(0, 1);
default_random_engine engine;

double parallel_part_time = 0.0;

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

double** lup_decomposition(double** A, int n, int t)
{
    auto P = generate_identity_matrix(n);

    for (auto i = 0; i < n; i++)
    {
        auto row = i;
        auto max = 0.0;

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
            auto start = omp_get_wtime();
            # pragma omp parallel for num_threads(t)
            for (auto q = 0; q < n; q++)
            {
                auto tmp = P[i][q];
                P[i][q] = P[row][q];
                P[row][q] = tmp;
                tmp = A[i][q];
                A[i][q] = A[row][q];
                A[row][q] = tmp;
            }
            auto elapsed = omp_get_wtime() - start;
            parallel_part_time += elapsed;
        }

        auto start = omp_get_wtime();
        # pragma omp parallel for num_threads(t)
        for (auto j = i + 1; j < n; j++)
        {
            A[j][i] /= A[i][i];

            for (auto k = i + 1; k < n; k++)
            {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
        auto elapsed = omp_get_wtime() - start;
        parallel_part_time += elapsed;
    }

    return P;
}

double** get_inverse(double** LU, double** P, int n, int t) {
    auto I = generate_zero_matrix(n);
    auto Y = generate_zero_matrix(n);

    auto start = omp_get_wtime();
    # pragma omp parallel for num_threads(t)
    for (auto q = 0; q < n; q++) 
    {
        for (auto i = 0; i < n; i++) 
        {
            Y[i][q] = P[i][q];
            auto j = 0;
            while (j < i) {
                Y[i][q] += -LU[i][j] * Y[j][q];
                j++;
            }
        }
        for (auto i = n - 1; i > -1; i--)
        {
            I[i][q] = Y[i][q];
            auto j = i + 1;
            while (j < n) 
            {
                I[i][q] += -LU[i][j] * I[j][q];
                j++;
            }
            I[i][q] = I[i][q] / LU[i][i];
        }
    }
    auto elapsed = omp_get_wtime() - start;
    parallel_part_time += elapsed;

    return I;
}

const int THREADS_NUM = 12;
const int RUNS_NUM = 3;
const int SEED = 73;
const int N = 1000;
const int MAX_N = 1000;
const int N_STEP = 100;

int main()
{
    auto steps = MAX_N / N_STEP;
    auto results1 = new double[steps];

    for (auto j = 0; j < steps; j++)
    {
        results1[j] = numeric_limits<double>::infinity();
    }

    for (auto j = 0; j < RUNS_NUM; j++)
    {
        engine.seed(SEED);

        for (auto i = 0; i < steps; i++)
        {
            auto n = (i + 1) * N_STEP;
            cout << n << "x" << n << " matrix" << endl;
            auto A = generate_random_matrix(n);
            auto start = omp_get_wtime();
            auto P = lup_decomposition(A, n, THREADS_NUM);
            auto I = get_inverse(A, P, n, THREADS_NUM);
            auto elapsed = omp_get_wtime() - start;
            if (elapsed < results1[i])
            {
                results1[i] = elapsed;
            }
            cout << "Elapsed(ms)=" << elapsed << endl;
        }

        cout << "Round finished" << endl;
    }

    cout << "Summary: " << endl;

    for (auto i = 0; i < steps; i++)
    {
        auto n = (i + 1) * N_STEP;
        cout << n << "x" << n << " matrix : ";
        cout << "Elapsed(ms)=" << results1[i] << endl;
    }

    auto results2 = new double[THREADS_NUM];
    auto min_parallel_time = numeric_limits<double>::infinity();

    for (auto j = 0; j < THREADS_NUM; j++)
    {
        results2[j] = numeric_limits<double>::infinity();
        min_parallel_time = numeric_limits<double>::infinity();
    }

    for (auto j = 0; j < RUNS_NUM; j++)
    {
        for (auto i = 1; i <= THREADS_NUM; i++)
        {
            parallel_part_time = 0.0;
            cout << i << " threads" << endl;
            auto A = generate_random_matrix(N);
            auto start = omp_get_wtime();
            auto P = lup_decomposition(A, N, i);
            auto I = get_inverse(A, P, N, i);
            auto elapsed = omp_get_wtime() - start;  
            if (elapsed < results2[i - 1])
            {
                results2[i - 1] = elapsed;
            }
            if (i == 1)
            {
                if (parallel_part_time < min_parallel_time)
                {
                    min_parallel_time = parallel_part_time;
                }
            }

            cout << "Elapsed(ms)=" << elapsed << endl;
        }

        cout << "Round finished" << endl;
    }

    cout << "Summary: " << endl;

    for (auto j = 0; j < THREADS_NUM; j++)
    {
        cout << j << " threads : ";
        cout << "Elapsed(ms)=" << results2[j] << endl;
    }

    cout << "Parallel part time: " << min_parallel_time;

    return 0;
}
