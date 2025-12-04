#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX 1000
int visited[MAX];

void dfs_parallel(int adj[MAX][MAX], int n, int node)
{
    visited[node] = 1;
    printf("%d ", node);

#pragma omp parallel for shared(adj, visited)
    for (int i = 0; i < n; i++)
    {
        if (adj[node][i] == 1 && !visited[i])
        {
#pragma omp task
            dfs_parallel(adj, n, i);
        }
    }
}

int main()
{
    int n = 5;
    int adj[MAX][MAX] = {
        {0, 1, 1, 0, 0},
        {1, 0, 1, 1, 0},
        {1, 1, 0, 0, 1},
        {0, 1, 0, 0, 1},
        {0, 0, 1, 1, 0}};

    double start = omp_get_wtime();

    for (int i = 0; i < n; i++)
        visited[i] = 0;

#pragma omp parallel
    {
#pragma omp single
        dfs_parallel(adj, n, 0);
    }

    double end = omp_get_wtime();

    printf("\nDFS Time: %f seconds\n", end - start);

    return 0;
}
