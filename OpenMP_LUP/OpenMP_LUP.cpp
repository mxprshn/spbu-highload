#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

double** generate_random_matrix(int n)
{
    double lower_bound = 0;
    double upper_bound = 1;
    uniform_real_distribution<double> unif(lower_bound, upper_bound);
    default_random_engine engine;

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
        }

        # pragma omp parallel for num_threads(t)
        for (auto j = i + 1; j < n; j++)
        {
            A[j][i] /= A[i][i];

            for (auto k = i + 1; k < n; k++)
            {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }

    return P;
}

//void print(double** matrix)
//{
//    int i = 0, j = 0;
//    for (i = 0; i < n; i++)
//    {
//        for (j = 0; j < n; j++)
//        {
//            cout << matrix[i][j] << '\t';
//        }
//        cout << endl;
//    }
//}

double** get_inverse(double** LU, double** P, int n, int t) {
    auto I = generate_zero_matrix(n);
    auto Y = generate_zero_matrix(n);

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

    return I;
}

int main()
{
    auto n = 1000;
    for (auto i = 1; i <= 12; i++)
    {
        cout << i << " threads\n";
        auto A = generate_random_matrix(n);
        auto start = chrono::steady_clock::now();
        auto P = lup_decomposition(A, n, i);
        auto I = get_inverse(A, P, n, i);
        cout << "Elapsed(ms)=" << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << endl;
    }

    return 0;
}
