#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

void initialize_array(double *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = (double)rand() / RAND_MAX * 100.0;
    }
}

/* ПОСЛЕДОВАТЕЛЬНЫЕ ОПЕРАЦИИ */
void sequential_operations(double *a, double *b, double *result_add, double *result_sub,
                           double *result_mul, double *result_div, int size)
{
    for (int i = 0; i < size; i++)
    {
        result_add[i] = a[i] + b[i];
        result_sub[i] = a[i] - b[i];
        result_mul[i] = a[i] * b[i];
        result_div[i] = (b[i] != 0.0) ? a[i] / b[i] : 0.0;
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    int n = 100000;
    int dimensions[2] = {0, 0};

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            n = atoi(argv[i + 1]);
            i++;
        }
    }

    dimensions[0] = (int)sqrt(n);
    while (n % dimensions[0] != 0)
    {
        dimensions[0]--;
    }
    dimensions[1] = n / dimensions[0];
    n = dimensions[0] * dimensions[1];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *a = NULL;
    double *b = NULL;
    double *result_add = NULL;
    double *result_sub = NULL;
    double *result_mul = NULL;
    double *result_div = NULL;

    double start_time, end_time;
    double parallel_time, sequential_time;

    if (rank == 0)
    {
        a = (double *)malloc(n * sizeof(double));
        b = (double *)malloc(n * sizeof(double));
        result_add = (double *)malloc(n * sizeof(double));
        result_sub = (double *)malloc(n * sizeof(double));
        result_mul = (double *)malloc(n * sizeof(double));
        result_div = (double *)malloc(n * sizeof(double));

        srand(time(NULL));
        initialize_array(a, n);
        initialize_array(b, n);

        /* НАЧАЛО ПОСЛЕДОВАТЕЛЬНОГО РАСЧЕТА */
        start_time = MPI_Wtime();
        sequential_operations(a, b, result_add, result_sub, result_mul, result_div, n);
        end_time = MPI_Wtime();
        sequential_time = end_time - start_time;
        /* КОНЕЦ ПОСЛЕДОВАТЕЛЬНОГО РАСЧЕТА */
    }

    MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);
    n = dimensions[0] * dimensions[1];

    int chunk_size = n / size;
    int remainder = n % size;

    int *send_counts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        send_counts[i] = chunk_size + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : (displs[i - 1] + send_counts[i - 1]);
    }

    double *local_a = (double *)malloc(send_counts[rank] * sizeof(double));
    double *local_b = (double *)malloc(send_counts[rank] * sizeof(double));
    double *local_add = (double *)malloc(send_counts[rank] * sizeof(double));
    double *local_sub = (double *)malloc(send_counts[rank] * sizeof(double));
    double *local_mul = (double *)malloc(send_counts[rank] * sizeof(double));
    double *local_div = (double *)malloc(send_counts[rank] * sizeof(double));

    /* НАЧАЛО ПАРАЛЛЕЛЬНОГО РАСЧЕТА - РАСПРЕДЕЛЕНИЕ ДАННЫХ */
    MPI_Scatterv(a, send_counts, displs, MPI_DOUBLE, local_a, send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, send_counts, displs, MPI_DOUBLE, local_b, send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    /* ПАРАЛЛЕЛЬНЫЕ ВЫЧИСЛЕНИЯ НА КАЖДОМ ПРОЦЕССЕ */
    for (int i = 0; i < send_counts[rank]; i++)
    {
        local_add[i] = local_a[i] + local_b[i];
        local_sub[i] = local_a[i] - local_b[i];
        local_mul[i] = local_a[i] * local_b[i];
        local_div[i] = (local_b[i] != 0.0) ? local_a[i] / local_b[i] : 0.0;
    }

    /* СБОР РЕЗУЛЬТАТОВ */
    MPI_Gatherv(local_add, send_counts[rank], MPI_DOUBLE, result_add, send_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_sub, send_counts[rank], MPI_DOUBLE, result_sub, send_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_mul, send_counts[rank], MPI_DOUBLE, result_mul, send_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_div, send_counts[rank], MPI_DOUBLE, result_div, send_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    parallel_time = end_time - start_time;
    /* КОНЕЦ ПАРАЛЛЕЛЬНОГО РАСЧЕТА */

    if (rank == 0)
    {
        printf("Parallel time: %.5f seconds\n", parallel_time);
        printf("Sequential time: %.5f seconds\n", sequential_time);

        free(a);
        free(b);
        free(result_add);
        free(result_sub);
        free(result_mul);
        free(result_div);
    }

    free(local_a);
    free(local_b);
    free(local_add);
    free(local_sub);
    free(local_mul);
    free(local_div);
    free(send_counts);
    free(displs);

    MPI_Finalize();
    return 0;
}