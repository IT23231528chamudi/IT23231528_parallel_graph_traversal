#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX 100

int visited[MAX];

// SERIAL BFS
void bfs_serial(int adj[MAX][MAX], int n, int start)
{
    int queue[MAX];
    int front = 0, rear = 0;

    for (int i = 0; i < n; i++)
        visited[i] = 0;

    visited[start] = 1;
    queue[rear++] = start;

    while (front < rear)
    {
        int current = queue[front++];

        for (int j = 0; j < n; j++)
        {
            if (adj[current][j] == 1 && !visited[j])
            {
                visited[j] = 1;
                queue[rear++] = j;
            }
        }
    }
}

// SERIAL DFS
void dfs_serial(int adj[MAX][MAX], int n, int node)
{
    visited[node] = 1;

    for (int i = 0; i < n; i++)
    {
        if (adj[node][i] == 1 && !visited[i])
        {
            dfs_serial(adj, n, i);
        }
    }
}

// MAIN FOR TIMING
int main()
{
    int n = 5;
    int adj[MAX][MAX] =
        {
            {0, 1, 1, 0, 0},
            {1, 0, 1, 1, 0},
            {1, 1, 0, 0, 1},
            {0, 1, 0, 0, 1},
            {0, 0, 1, 1, 0}};

    clock_t start, end;

    // BFS TIME
    start = clock();
    bfs_serial(adj, n, 0);
    end = clock();
    double bfs_time = (double)(end - start) / CLOCKS_PER_SEC;

    // DFS TIME
    for (int i = 0; i < n; i++)
        visited[i] = 0;
    start = clock();
    dfs_serial(adj, n, 0);
    end = clock();
    double dfs_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nSerial BFS Time: %f seconds\n", bfs_time);
    printf("Serial DFS Time: %f seconds\n", dfs_time);

    return 0;
}
