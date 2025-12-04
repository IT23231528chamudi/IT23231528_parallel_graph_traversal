#include <mpi.h>
#include <stdio.h>

#define MAX 100

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    int adj[MAX][MAX] = {
        {0, 1, 1, 0, 0},
        {1, 0, 1, 1, 0},
        {1, 1, 0, 0, 1},
        {0, 1, 0, 0, 1},
        {0, 0, 1, 1, 0}};

    int visited[MAX] = {0};
    int n = 5, start = 0;

    if (rank == 0)
    {
        printf("MPI BFS Traversal:\n");
    }

    for (int i = rank; i < n; i += size)
    {
        if (adj[start][i] == 1 && visited[i] == 0)
        {
            visited[i] = 1;
            printf("%d from process %d\n", i, rank);
        }
    }

    double end_time = MPI_Wtime();

    if (rank == 0)
        printf("\nMPI BFS Time (%d processes): %f seconds\n", size, end_time - start_time);

    MPI_Finalize();
    return 0;
}
