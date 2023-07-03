// MP 1
#include "../wb.h"

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
    // @@ Insert code to implement vector addition here
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < len) out[n] = in1[n] + in2[n]; 
}

int main(int argc, char **argv) {
    wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = 
        (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = 
        (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);

    hostOutput = (float*)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");
    // @@ Allocate GPU memory here
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, inputLength * sizeof(float));
    cudaMalloc((void **) &d_B, inputLength * sizeof(float));
    cudaMalloc((void **) &d_C, inputLength * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    // @@ Copy memory to the GPU here
    cudaMemcpy(d_A, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    // @@ Initialize the grid and block dimensions here
    // const int block_size = 256;
    // const int grid_size = inputLength / block_size + 1;
    dim3 grid(ceil((float)inputLength / 256), 1, 1);
    dim3 block(256, 1, 1);
    wbTime_start(Compute, "Performing CUDA computation.");
    vecAdd<<<grid, block>>>(d_A, d_B, d_C, inputLength);
    // @@ Launch the GPU Kernel here
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation.");

    wbTime_start(Copy, "Copying output memory to the CPU.");
    // @@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, d_C, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU.");

    wbTime_start(GPU, "Freeing GPU memory");
    // @@ Free the GPU memory here
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    wbTime_stop(GPU, "Freeing GPU memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}