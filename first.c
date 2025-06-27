#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);  // Инициализация MPI в начале
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = 100000;  // Значение по умолчанию
    double* array = NULL;
    double sum_seq = 0.0;
    double seq_time = 0.0;

    // Обработка аргументов только в процессе 0
    if (world_rank == 0) {
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

        // Генерация данных только в процессе 0
        array = malloc(N * sizeof(double));
        srand((unsigned)time(NULL) ^ getpid());
        for (int i = 0; i < N; ++i) {
            array[i] = (double)rand() / RAND_MAX;
        }

        // Последовательный расчёт
        double start = MPI_Wtime();
        for (int i = 0; i < N; ++i) sum_seq += array[i];
        seq_time = MPI_Wtime() - start;
        printf("Sequential time: %.6f sec\n", seq_time);
    }

    // Широковещательная передача N всем процессам
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Вычисление локального размера
    int base = N / world_size;
    int rem = N % world_size;
    int local_n = (world_rank < rem) ? base + 1 : base;

    // Вычисление counts/displs локально (без MPI_Bcast)
    int* counts = malloc(world_size * sizeof(int));
    int* displs = malloc(world_size * sizeof(int));
    int offset = 0;
    for (int p = 0; p < world_size; ++p) {
        counts[p] = (p < rem) ? base + 1 : base;
        displs[p] = offset;
        offset += counts[p];
    }

    // Выделение памяти для локальных данных (с проверкой local_n)
    double* local = NULL;
    if (local_n > 0) {
        local = malloc(local_n * sizeof(double));
        if (!local) {
            perror("malloc");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Распределение данных
    MPI_Scatterv(
        array, counts, displs, MPI_DOUBLE,
        local, local_n, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Параллельное суммирование
    MPI_Barrier(MPI_COMM_WORLD);
    double par_start = MPI_Wtime();
    
    double local_sum = 0.0;
    for (int i = 0; i < local_n; ++i) local_sum += local[i];
    
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double par_time = MPI_Wtime() - par_start;

    // Вывод результатов в процессе 0
    if (world_rank == 0) {
        printf("Parallel time:   %.6f sec\n", par_time);
        printf("Speedup:         %.2f\n", seq_time / par_time);
        
        // Верификация результатов
        double diff = global_sum - sum_seq;
        if (diff < 0) diff = -diff;
        if (diff < 1e-8) {
            printf("Result: CORRECT\n");
        } else {
            printf("Result: ERROR (diff=%.10f)\n", diff);
        }
    }

    // Освобождение памяти
    if (world_rank == 0) free(array);
    if (local_n > 0) free(local);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}