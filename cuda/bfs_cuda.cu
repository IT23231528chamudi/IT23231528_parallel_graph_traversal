#include <stdio.h>
#include <cuda_runtime.h>

#define N 5 // small test graph

// CUDA BFS Kernel
__global__ void bfs_kernel(int *adj, int *visited, int *queue, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n && queue[tid] == 1)
    {
        for (int j = 0; j < n; j++)
        {
            if (adj[tid * n + j] == 1 && visited[j] == 0)
            {
                visited[j] = 1;
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
    int init_queue[N] = {0};
    init_visited[0] = 1;
    init_queue[0] = 1;

    int blocks_list[] = {1, 2, 4};
    int threads_list[] = {1, 2, 4, 8, 16, 32, 64};

    int *d_adj, *d_visited, *d_queue;

    cudaMalloc(&d_adj, N * N * sizeof(int));
    cudaMalloc(&d_visited, N * sizeof(int));
    cudaMalloc(&d_queue, N * sizeof(int));

    cudaMemcpy(d_adj, adj, N * N * sizeof(int), cudaMemcpyHostToDevice);

    printf("\n===== CUDA BFS Full Evaluation =====\n");

    for (int b = 0; b < 3; b++)
    {
        for (int t = 0; t < 7; t++)
        {
            int blocks = blocks_list[b];
            int threads = threads_list[t];

            cudaMemcpy(d_visited, init_visited, N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_queue, init_queue, N * sizeof(int), cudaMemcpyHostToDevice);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            bfs_kernel<<<blocks, threads>>>(d_adj, d_visited, d_queue, N);
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
    cudaFree(d_queue);

    return 0;
}
