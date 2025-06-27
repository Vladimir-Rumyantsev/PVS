#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>
#include <math.h>

#define EPSILON 1e-10

typedef struct {
    double* sum;
    double* diff;
    double* prod;
    double* quot;
} OperationResults;

void compute_operations(double* a, double* b, OperationResults* res, int n) {
    for (int i = 0; i < n; i++) {
        res->sum[i] = a[i] + b[i];
        res->diff[i] = a[i] - b[i];
        res->prod[i] = a[i] * b[i];
        res->quot[i] = (fabs(b[i]) > EPSILON) ? a[i] / b[i] : 0.0;
    }
}

int compare_results(OperationResults* res1, OperationResults* res2, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(res1->sum[i] - res2->sum[i]) > EPSILON) return 0;
        if (fabs(res1->diff[i] - res2->diff[i]) > EPSILON) return 0;
        if (fabs(res1->prod[i] - res2->prod[i]) > EPSILON) return 0;
        if (fabs(res1->quot[i] - res2->quot[i]) > EPSILON) return 0;
    }
    return 1;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 10000000;
    double *a = NULL, *b = NULL;
    OperationResults seq_res = {0}, par_res = {0};
    double seq_time = 0.0;

    // Обработка аргументов только в root
    if (rank == 0) {
        int opt;
        while ((opt = getopt(argc, argv, "n:")) != -1) {
            if (opt == 'n') {
                N = atoi(optarg);
                if (N <= 0) {
                    fprintf(stderr, "Error: N must be positive\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
            } else {
                fprintf(stderr, "Usage: %s [-n size]\n", argv[0]);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        // Выделение памяти и инициализация
        a = malloc(N * sizeof(double));
        b = malloc(N * sizeof(double));
        seq_res.sum = malloc(N * sizeof(double));
        seq_res.diff = malloc(N * sizeof(double));
        seq_res.prod = malloc(N * sizeof(double));
        seq_res.quot = malloc(N * sizeof(double));
        par_res.sum = malloc(N * sizeof(double));
        par_res.diff = malloc(N * sizeof(double));
        par_res.prod = malloc(N * sizeof(double));
        par_res.quot = malloc(N * sizeof(double));

        if (!a || !b || !seq_res.sum || !seq_res.diff || !seq_res.prod || !seq_res.quot ||
            !par_res.sum || !par_res.diff || !par_res.prod || !par_res.quot) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Генерация данных (без нулей в знаменателе)
        srand(time(NULL) ^ getpid());
        for (int i = 0; i < N; i++) {
            a[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
            b[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        }

        // Последовательное выполнение
        double start = MPI_Wtime();
        compute_operations(a, b, &seq_res, N);
        seq_time = MPI_Wtime() - start;
        printf("Sequential time: %.5f sec\n", seq_time);
    }

    // Широковещательная передача N
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Вычисление распределения данных
    int base_size = N / size;
    int remainder = N % size;
    int local_size = base_size + (rank < remainder ? 1 : 0);

    int* counts = malloc(size * sizeof(int));
    int* displs = malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        counts[i] = base_size + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }

    // Выделение памяти для локальных данных
    double* local_a = malloc(local_size * sizeof(double));
    double* local_b = malloc(local_size * sizeof(double));
    OperationResults local_res = {
        .sum = malloc(local_size * sizeof(double)),
        .diff = malloc(local_size * sizeof(double)),
        .prod = malloc(local_size * sizeof(double)),
        .quot = malloc(local_size * sizeof(double))
    };

    if (!local_a || !local_b || !local_res.sum || !local_res.diff || 
        !local_res.prod || !local_res.quot) {
        fprintf(stderr, "[%d] Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Синхронизация перед началом измерений
    MPI_Barrier(MPI_COMM_WORLD);
    double par_start = MPI_Wtime();

    // Распределение данных
    MPI_Scatterv(a, counts, displs, MPI_DOUBLE, local_a, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, counts, displs, MPI_DOUBLE, local_b, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Параллельные вычисления
    compute_operations(local_a, local_b, &local_res, local_size);

    // Сбор результатов
    MPI_Gatherv(local_res.sum, local_size, MPI_DOUBLE, par_res.sum, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_res.diff, local_size, MPI_DOUBLE, par_res.diff, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_res.prod, local_size, MPI_DOUBLE, par_res.prod, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_res.quot, local_size, MPI_DOUBLE, par_res.quot, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double par_time = MPI_Wtime() - par_start;

    // Освобождение локальной памяти
    free(local_a);
    free(local_b);
    free(local_res.sum);
    free(local_res.diff);
    free(local_res.prod);
    free(local_res.quot);
    free(counts);
    free(displs);

    // Проверка результатов и вывод
    if (rank == 0) {
        printf("Parallel time:   %.5f sec\n", par_time);
        printf("Speedup:         %.2f\n", seq_time / par_time);
        
        if (compare_results(&seq_res, &par_res, N)) {
            printf("Result: CORRECT\n");
        } else {
            printf("Result: INCORRECT\n");
        }

        free(a);
        free(b);
        free(seq_res.sum);
        free(seq_res.diff);
        free(seq_res.prod);
        free(seq_res.quot);
        free(par_res.sum);
        free(par_res.diff);
        free(par_res.prod);
        free(par_res.quot);
    }

    MPI_Finalize();
    return 0;
}