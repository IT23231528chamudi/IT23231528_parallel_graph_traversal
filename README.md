# Parallel Graph Traversal (BFS & DFS)

This repository contains implementations of Breadth-First Search (BFS) and Depth-First Search (DFS) on graphs using several parallelization approaches and a serial baseline for performance comparison.

Directory layout
- `serial/` – serial reference implementation and `Makefile`.
- `openmp/` – OpenMP implementations (`bfs_openmp.c`, `dfs_openmp.c`) and `Makefile`.
- `mpi/` – MPI implementations (`bfs_mpi.c`, `dfs_mpi.c`) and `Makefile`.
- `cuda/` – CUDA implementations (`bfs_cuda.cu`, `dfs_cuda.cu`) and `Makefile`.

Prerequisites
- A POSIX-like build environment (Linux, macOS, or WSL on Windows) with `make` and a C compiler (`gcc`/`clang`).
- For OpenMP: a compiler with OpenMP support (e.g. `gcc` >= 4.9).
- For MPI: Open MPI or MPICH installed (`mpicc`, `mpirun`/`mpiexec`).
- For CUDA: NVIDIA GPU and CUDA toolkit installed (`nvcc`).

Notes on Windows
If you are on Windows, it is recommended to use WSL2 or a Linux environment to build and run the MPI/CUDA/OpenMP workflows reliably. Native Windows builds may work if you have compatible toolchains (MSYS2/MinGW, MS MPI, CUDA for Windows), but commands below assume a Unix-style shell.

Building
Each subfolder contains a `Makefile` to build the relevant binaries. Examples below assume you're in the repository root.

Build the serial implementation:
```powershell
cd serial; make
```

Build the OpenMP implementations:
```powershell
cd openmp; make
```

Build the MPI implementations:
```powershell
cd mpi; make
```

Build the CUDA implementations (requires `nvcc`):
```powershell
cd cuda; make
```

Running
General notes:
- The programs expect graph inputs from the `datasets/` folder (or other graph files). Check each program's usage or source for exact command-line arguments.
- On Windows PowerShell, run executables with `.inaryName` (for example `.fs_openmp`). On WSL or Linux, use `./binaryName`.

Serial example:
```powershell
cd serial
./serial 
```

OpenMP example (set number of threads via `OMP_NUM_THREADS`):
```powershell
cd openmp
$env:OMP_NUM_THREADS=4;
./bfs_openmp  
```

MPI example (run with 4 processes):
```powershell
cd mpi
mpirun -np 4
./bfs_mpi  
```

CUDA example (if built and a CUDA-capable GPU is present):
```powershell
cd cuda
%%writefile bfs_cuda.cu
!nvcc bfs_cuda.cu -o bfs_cuda
!./bfs_cuda

%%writefile dfs_cuda.cu
!nvcc dfs_cuda.cu -o dfs_cuda
!./dfs_cuda
```
 

 

 
