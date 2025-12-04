#include <stdio.h>
#include <cuda_runtime.h>

#define N 5 // small test graph

// CUDA DFS Kernel (only thread 0 executes DFS)
__global__ void dfs_kernel(int *adj, int *visited, int *stack, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        int top = -1;

        stack[++top] = 0;
        visited[0] = 1;

        while (top >= 0)
        {
            int node = stack[top--];

            for (int j = n - 1; j >= 0; j--)
            {
                if (adj[node * n + j] == 1 && visited[j] == 0)
                {
                    visited[j] = 1;
                    stack[++top] = j;
                }
            }
        }
    }
}

int main()
{
    int adj[N][N] = {
        {0, 1, 1, 0, 0},
        {1, 0, 1, 1, 0},
        {1, 1, 0, 0, 1},
        {0, 1, 0, 0, 1},
        {0, 0, 1, 1, 0}};

    int init_visited[N] = {0};
    int init_stack[N] = {0};

    int blocks_list[] = {1, 2, 4};
    int threads_list[] = {1, 2, 4, 8, 16, 32, 64};

    int *d_adj, *d_visited, *d_stack;

    cudaMalloc(&d_adj, N * N * sizeof(int));
    cudaMalloc(&d_visited, N * sizeof(int));
    cudaMalloc(&d_stack, N * sizeof(int));

    cudaMemcpy(d_adj, adj, N * N * sizeof(int), cudaMemcpyHostToDevice);

    printf("\n===== CUDA DFS Full Evaluation =====\n");

    for (int b = 0; b < 3; b++)
    {
        for (int t = 0; t < 7; t++)
        {
            int blocks = blocks_list[b];
            int threads = threads_list[t];

            cudaMemcpy(d_visited, init_visited, N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_stack, init_stack, N * sizeof(int), cudaMemcpyHostToDevice);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            dfs_kernel<<<blocks, threads>>>(d_adj, d_visited, d_stack, N);
            cudaDeviceSynchronize();

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);

            int visited[N];
            cudaMemcpy(visited, d_visited, N * sizeof(int), cudaMemcpyDeviceToHost);

            printf("\nBlocks = %d  Threads/Block = %d\n", blocks, threads);
            printf("Visited: ");
            for (int i = 0; i < N; i++)
                if (visited[i])
                    printf("%d ", i);
            printf("\nTime: %f ms\n", ms);
        }
    }

    cudaFree(d_adj);
    cudaFree(d_visited);
    cudaFree(d_stack);

    return 0;
}
