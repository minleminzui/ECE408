#include "../wb.h"

#define TILE_SIZE 32
#define wbCheck(stmt) \
    do { \
        cudaError_t err = stmt; \
        if (err != cudaSuccess) { \
            wbLog(ERROR, "Failed to run stmt ", #stmt); \
            wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err)) \
            return -1; \
        }\
    } while(0)\

// Compute C = A * B
// Tiled
__global__ void matrixMultiply(float *A, float *B, float *C, 
                                int numARows, int numAColumns,
                                int numBRows, int numBColumns,
                                int numCRows, int numCColumns) 
{
    // use shared memory
    __shared__ float Mds[TILE_SIZE][TILE_SIZE];
    __shared__ float Nds[TILE_SIZE][TILE_SIZE];


    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    float Pvalue = 0;

    for(int m = 0; m < (numBRows - 1) / TILE_SIZE + 1; ++m) {
        
        if (Row < numARows && m * TILE_SIZE + tx < numAColumns) {
            Mds[ty][tx] = A[Row * numAColumns + m * TILE_SIZE + tx];
        } else {
            Mds[ty][tx] = 0;
        }
        if (Col < numBColumns && m * TILE_SIZE + ty < numBRows) {
            Nds[ty][tx] = B[(m * TILE_SIZE + ty) * numBColumns + Col];
        } else {
            Nds[ty][tx] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    if (Row < numCRows && Col < numCColumns) {
        C[Row * numCColumns + Col] = Pvalue;
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostA;
    float *hostB;
    float *hostC;
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numsARows;
    int numsAColumns;
    int numsBRows;
    int numsBColumns;
    int numsCRows;
    int numsCColumns;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");

    hostA = (float*)wbImport(wbArg_getInputFile(args, 0), &numsARows, &numsAColumns);
    hostB = (float*)wbImport(wbArg_getInputFile(args, 1), &numsBRows, &numsBColumns);

    // set numsCRows and numCColumns
    numsCRows = numsARows;
    numsCColumns = numsBColumns;
    // allocate the hostC matrix
    hostC = (float*)malloc(sizeof(float) * numsCRows * numsCColumns);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimension of A are ", numsARows, " x ", numsAColumns);
    wbLog(TRACE, "The dimension of A are ", numsBRows, " x ", numsBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    // allocate GPU memory here
    cudaMalloc((void **)&deviceA, numsARows * numsAColumns * sizeof(float));
    cudaMalloc((void **)&deviceB, numsBRows * numsBColumns * sizeof(float));
    cudaMalloc((void **)&deviceC, numsCRows * numsCColumns * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU,");
    // Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeof(float) * numsARows * numsAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float) * numsBRows * numsBColumns, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU,");

    // Initailize the grid and block dimension here
    dim3 Grid(ceil((float)numsCColumns / TILE_SIZE), ceil((float)numsCRows / TILE_SIZE), 1);
    dim3 Block(TILE_SIZE, TILE_SIZE, 1);
    wbTime_start(Compute, "Performing CUDA computation");
    // Launch the GPU Kernel here
    matrixMultiply<<<Grid, Block>>>(deviceA, deviceB, deviceC, numsARows, numsAColumns,
                                    numsBRows, numsCColumns, numsCRows, numsCColumns);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation.");

    wbTime_start(Copy, "Copying output memory to the CPU");
    // Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeof(float) * numsCRows * numsCColumns, cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    // Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");
    
    wbSolution(args, hostC, numsCRows, numsCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}