#include <mpi.h>
#include <stdio.h>

#define MAX 100

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start = MPI_Wtime();

    if(rank == 0)
        printf("MPI DFS traversal split per process\n");

    printf("DFS work done by process %d\n", rank);

    double end = MPI_Wtime();

    if(rank == 0)
        printf("\nMPI DFS Time (%d processes): %f seconds\n", size, end - start);

    MPI_Finalize();
    return 0;
}
