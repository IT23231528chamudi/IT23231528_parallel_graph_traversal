#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX 100

// PARALLEL BFS
double bfs_parallel(int adj[MAX][MAX], int n, int start)
{
    int visited[MAX] = {0};
    int queue[MAX];
    int front = 0, rear = 0;

    int num_threads = omp_get_max_threads(); // read from OMP_NUM_THREADS

    double start_time = omp_get_wtime();

    queue[rear++] = start;
    visited[start] = 1;

    printf("Parallel BFS Traversal (threads = %d): ", num_threads);

    while (front < rear)
    {
        int level_size = rear - front; //level-synchronous parallel BFS

#pragma omp parallel for shared(queue, visited)
        for (int i = front; i < front + level_size; i++)
        {
            int current = queue[i];
            printf("%d ", current);

            for (int j = 0; j < n; j++)
            {
                if (adj[current][j] == 1 && !visited[j])
                {
                    visited[j] = 1;
                    queue[rear++] = j;
                }
            }
        }

        front += level_size;
    }

    double end_time = omp_get_wtime();
    double exec_time = end_time - start_time;

    printf("\nTime: %f seconds\n", exec_time);

    return exec_time;
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

    bfs_parallel(adj, n, 0);

    return 0;
}
