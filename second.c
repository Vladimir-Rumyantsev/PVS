#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

void bubble_sort(double *arr, int n) {
    for (int i = 0; i < n - 1; ++i) {
        int swapped = 0;
        for (int j = 0; j < n - 1 - i; ++j) {
            if (arr[j] > arr[j+1]) {
                double tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
                swapped = 1;
            }
        }
        if (!swapped) break;
    }
}

void merge_sorted_arrays(double *merged, double *parts, int *counts, int *displs, int world_size) {
    int *idx = calloc(world_size, sizeof(int));
    for (int i = 0; i < displs[world_size-1] + counts[world_size-1]; ++i) {
        int best_p = -1;
        double best_val;
        
        for (int p = 0; p < world_size; p++) {
            if (idx[p] >= counts[p]) continue;
            
            double val = parts[displs[p] + idx[p]];
            if (best_p == -1 || val < best_val) {
                best_p = p;
                best_val = val;
            }
        }
        merged[i] = best_val;
        idx[best_p]++;
    }
    free(idx);
}

int is_sorted(double *arr, int n) {
    for (int i = 0; i < n-1; i++)
        if (arr[i] > arr[i+1]) return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = 100000;
    double *arr_seq = NULL;
    double *arr_par = NULL;
    double seq_time = 0.0;

    // Process arguments only in root
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

        // Generate data
        arr_seq = malloc(N * sizeof(double));
        arr_par = malloc(N * sizeof(double));
        srand(time(NULL) ^ getpid());
        for (int i = 0; i < N; ++i) {
            arr_seq[i] = arr_par[i] = (double)rand() / RAND_MAX;
        }

        // Sequential sort
        double start = MPI_Wtime();
        bubble_sort(arr_seq, N);
        seq_time = MPI_Wtime() - start;
        printf("Sequential time: %.6f sec\n", seq_time);
        printf("Seq sorted: %s\n", is_sorted(arr_seq, N) ? "YES" : "NO");
    }

    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate local sizes
    int base = N / world_size;
    int rem = N % world_size;
    int local_n = (world_rank < rem) ? base + 1 : base;

    // Compute counts/displs locally
    int *counts = malloc(world_size * sizeof(int));
    int *displs = malloc(world_size * sizeof(int));
    int offset = 0;
    for (int p = 0; p < world_size; p++) {
        counts[p] = (p < rem) ? base + 1 : base;
        displs[p] = offset;
        offset += counts[p];
    }

    // Allocate local array
    double *local = malloc(local_n * sizeof(double));

    // Start parallel timing
    MPI_Barrier(MPI_COMM_WORLD);
    double par_start = MPI_Wtime();

    // Distribute data
    MPI_Scatterv(
        arr_par, counts, displs, MPI_DOUBLE,
        local, local_n, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Local sort
    bubble_sort(local, local_n);

    // Gather sorted parts
    MPI_Gatherv(
        local, local_n, MPI_DOUBLE,
        arr_par, counts, displs, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Merge sorted parts in root
    double *merged = NULL;
    if (world_rank == 0) {
        merged = malloc(N * sizeof(double));
        merge_sorted_arrays(merged, arr_par, counts, displs, world_size);
    }

    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    double par_time = MPI_Wtime() - par_start;

    // Verification and output
    if (world_rank == 0) {
        printf("Parallel time:   %.6f sec\n", par_time);
        printf("Speedup:         %.2f\n", seq_time / par_time);
        printf("Par sorted:      %s\n", is_sorted(merged, N) ? "YES" : "NO");
        
        // Verify against sequential
        int correct = 1;
        for (int i = 0; i < N; i++) {
            if (arr_seq[i] != merged[i]) {
                correct = 0;
                break;
            }
        }
        printf("Result:          %s\n", correct ? "CORRECT" : "INCORRECT");
    }

    // Cleanup
    free(local);
    free(counts);
    free(displs);
    if (world_rank == 0) {
        free(arr_seq);
        free(arr_par);
        free(merged);
    }

    MPI_Finalize();
    return 0;
}