#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define EPSILON 1e-10

typedef struct {
    double* add;
    double* sub;
    double* mul;
    double* div;
} MatrixResults;

void initialize_array(double *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
    }
}

void compute_operations(double *a, double *b, MatrixResults* res, int size) {
    for (int i = 0; i < size; i++) {
        res->add[i] = a[i] + b[i];
        res->sub[i] = a[i] - b[i];
        res->mul[i] = a[i] * b[i];
        res->div[i] = (fabs(b[i]) > EPSILON) ? a[i] / b[i] : 0.0;
    }
}

int compare_results(MatrixResults* res1, MatrixResults* res2, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(res1->add[i] - res2->add[i]) > EPSILON) return 0;
        if (fabs(res1->sub[i] - res2->sub[i]) > EPSILON) return 0;
        if (fabs(res1->mul[i] - res2->mul[i]) > EPSILON) return 0;
        if (fabs(res1->div[i] - res2->div[i]) > EPSILON) return 0;
    }
    return 1;
}

void find_optimal_dimensions(int n, int* rows, int* cols) {
    *rows = (int)sqrt(n);
    while (n % *rows != 0 && *rows > 1) {
        (*rows)--;
    }
    *cols = n / *rows;
    
    // Ensure we have at least 100000 elements
    while (*rows * *cols < 100000 && *rows > 1) {
        (*rows)--;
        *cols = n / *rows;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int original_n = 100000;
    int adjusted_n = original_n;
    int rows = 0, cols = 0;
    double *a = NULL, *b = NULL;
    MatrixResults seq_res = {0}, par_res = {0};
    double seq_time = 0.0;

    // Обработка аргументов только в root
    if (rank == 0) {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                original_n = atoi(argv[i+1]);
                if (original_n < 100000) {
                    fprintf(stderr, "Error: N must be at least 100000\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                i++;
            }
        }

        // Находим оптимальные размеры матрицы
        find_optimal_dimensions(original_n, &rows, &cols);
        adjusted_n = rows * cols;
        
        if (adjusted_n != original_n) {
            printf("Adjusted array size: %d -> %d (%d x %d)\n", 
                   original_n, adjusted_n, rows, cols);
        }

        // Выделение памяти
        a = malloc(adjusted_n * sizeof(double));
        b = malloc(adjusted_n * sizeof(double));
        seq_res.add = malloc(adjusted_n * sizeof(double));
        seq_res.sub = malloc(adjusted_n * sizeof(double));
        seq_res.mul = malloc(adjusted_n * sizeof(double));
        seq_res.div = malloc(adjusted_n * sizeof(double));
        par_res.add = malloc(adjusted_n * sizeof(double));
        par_res.sub = malloc(adjusted_n * sizeof(double));
        par_res.mul = malloc(adjusted_n * sizeof(double));
        par_res.div = malloc(adjusted_n * sizeof(double));

        if (!a || !b || !seq_res.add || !seq_res.sub || !seq_res.mul || !seq_res.div ||
            !par_res.add || !par_res.sub || !par_res.mul || !par_res.div) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Инициализация данных
        srand(time(NULL));
        initialize_array(a, adjusted_n);
        initialize_array(b, adjusted_n);

        // Последовательные вычисления
        double start = MPI_Wtime();
        compute_operations(a, b, &seq_res, adjusted_n);
        seq_time = MPI_Wtime() - start;
        printf("Sequential time: %.5f sec\n", seq_time);
    }

    // Рассылаем размеры матрицы
    int dimensions[2] = {rows, cols};
    MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&adjusted_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    rows = dimensions[0];
    cols = dimensions[1];

    // Вычисление распределения данных
    int base_chunk = adjusted_n / size;
    int remainder = adjusted_n % size;
    int local_size = base_chunk + (rank < remainder ? 1 : 0);

    int* send_counts = malloc(size * sizeof(int));
    int* displs = malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += send_counts[i];
    }

    // Выделение памяти для локальных данных
    double* local_a = malloc(local_size * sizeof(double));
    double* local_b = malloc(local_size * sizeof(double));
    MatrixResults local_res = {
        .add = malloc(local_size * sizeof(double)),
        .sub = malloc(local_size * sizeof(double)),
        .mul = malloc(local_size * sizeof(double)),
        .div = malloc(local_size * sizeof(double))
    };

    if (!local_a || !local_b || !local_res.add || !local_res.sub || 
        !local_res.mul || !local_res.div) {
        fprintf(stderr, "[%d] Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Синхронизация перед замерами
    MPI_Barrier(MPI_COMM_WORLD);
    double par_start = MPI_Wtime();

    // Распределение данных
    MPI_Scatterv(a, send_counts, displs, MPI_DOUBLE, local_a, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, send_counts, displs, MPI_DOUBLE, local_b, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Параллельные вычисления
    compute_operations(local_a, local_b, &local_res, local_size);

    // Сбор результатов
    MPI_Gatherv(local_res.add, local_size, MPI_DOUBLE, par_res.add, send_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_res.sub, local_size, MPI_DOUBLE, par_res.sub, send_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_res.mul, local_size, MPI_DOUBLE, par_res.mul, send_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_res.div, local_size, MPI_DOUBLE, par_res.div, send_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double par_time = MPI_Wtime() - par_start;

    // Освобождение локальной памяти
    free(local_a);
    free(local_b);
    free(local_res.add);
    free(local_res.sub);
    free(local_res.mul);
    free(local_res.div);
    free(send_counts);
    free(displs);

    // Проверка результатов и вывод
    if (rank == 0) {
        printf("Parallel time:   %.5f sec\n", par_time);
        printf("Speedup:         %.2f\n", seq_time / par_time);
        
        if (compare_results(&seq_res, &par_res, adjusted_n)) {
            printf("Result: CORRECT\n");
        } else {
            printf("Result: INCORRECT\n");
        }

        free(a);
        free(b);
        free(seq_res.add);
        free(seq_res.sub);
        free(seq_res.mul);
        free(seq_res.div);
        free(par_res.add);
        free(par_res.sub);
        free(par_res.mul);
        free(par_res.div);
    }

    MPI_Finalize();
    return 0;
}