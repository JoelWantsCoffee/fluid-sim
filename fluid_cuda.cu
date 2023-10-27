#include <stdio.h>
#include <vector_types.h>

#define flow density

#include "fluid.h"
#include "fluid_cuda.cuh"

void check_error(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        printf("CUDA error: %d : %s\n", int(e), cudaGetErrorString(e));
        abort();
    }
}

__global__ void multiply_into_simple(int N, int stride_a, const float* __restrict__ A, int stride_b, const float* __restrict__ B, int stride_c, float* __restrict__ C)
{
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float lilsum = 0;

    for (int k = 0; k < N; k += 32)
    {
        As[threadIdx.y][threadIdx.x] = A[j*stride_a + k + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y)*stride_b + i];
        __syncthreads();

        #pragma unroll
        for (int l = 0; l < 32; l++)
        {
            lilsum += As[threadIdx.y][l] * Bs[l][threadIdx.x];
        }
        __syncthreads();
    }

    C[j*stride_c + i] = lilsum;
}


__host__ void matrixMultiply_GPU(int N, const float* A, const float* B, float* C, int *arg, int argCount)
{
    float *A_device;
    float *B_device;
    float *C_device;
    // float *buffer_device;

    check_error(cudaMalloc(&A_device, N*N*sizeof(float)));
    check_error(cudaMalloc(&B_device, N*N*sizeof(float)));
    check_error(cudaMalloc(&C_device, N*N*sizeof(float)));
    // check_error(cudaMalloc(&buffer_device, sizeof(float) * buffer_required_GPU(N) ));

    check_error(cudaMemcpy(A_device, A, N*N*sizeof(float), cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(B_device, B, N*N*sizeof(float), cudaMemcpyHostToDevice));

    // check_error(cudaMemset(C_device, 0, N*N*sizeof(float)));
    // check_error(cudaMemset(buffer_device, 0, sizeof(float) * buffer_required_GPU(N) ));
    // check_error(cudaMemcpy(C_device, C, N*N*sizeof(float), cudaMemcpyHostToDevice));

    multiply_into_simple<<<dim3(2048 / 32, 2048 / 32, 1), dim3(32,32,1)>>>(N, N, B_device, N, A_device, N, C_device);
    // multiply_into_GPU(N, N, B_device, N, A_device, N, C_device, buffer_device);
    // matrixMultiplyKernel_GPU<<<1, dim3(shared_size,shared_size,1)>>>(N, B_device, A_device, C_device, 0, 0, 0);

    check_error(cudaPeekAtLastError());

    check_error(cudaDeviceSynchronize());

    check_error(cudaMemcpy(C, C_device, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    
    check_error(cudaFree(A_device));
    check_error(cudaFree(B_device));
    check_error(cudaFree(C_device));
    // check_error(cudaFree(buffer_device));

    return;
}

__global__ void project_block_simple(float * flow, float * from_velx, float * from_vely, float * to_velx, float * to_vely) 
{
    /*
    
    tu = t + 1i
    tv = t + 1j

    float s = t->density + t->density + tu->density + tv->density;
    
    s = !s ? 0 : (1 / s);

    float d = (t->vel_x + t->vel_y - tu->vel_x - tv->vel_y) * s;

    (t - from + into)->vel_x -= d * t->density;
    (t - from + into)->vel_y -= d * t->density;
    (tu - from + into)->vel_x += d * tu->density;
    (tv - from + into)->vel_y += d * tv->density;
    
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index = i + j * WIDTH;

    to_velx[index] = from_velx[index];
    to_vely[index] = from_vely[index];
}

__host__ void project_all_gpu(struct Tile * from, struct Tile * to)
{
    // Load memory onto the GPU
    float * flow;    check_error(cudaMalloc(&flow, WIDTH*HEIGHT*sizeof(float)));
    float * from_velx;  check_error(cudaMalloc(&from_velx, WIDTH*HEIGHT*sizeof(float)));
    float * from_vely;  check_error(cudaMalloc(&from_vely, WIDTH*HEIGHT*sizeof(float)));
    float * to_velx;    check_error(cudaMalloc(&to_velx, WIDTH*HEIGHT*sizeof(float)));
    float * to_vely;    check_error(cudaMalloc(&to_vely, WIDTH*HEIGHT*sizeof(float)));

    for (int j = 0; j < HEIGHT; j++)
    for (int i = 0; i < WIDTH; i++)
    {
        int index = i + j * WIDTH;
        flow[index] = from[index].flow;
        from_velx[index] = from[index].vel_x;
        from_vely[index] = from[index].vel_y;
        to_velx[index] = 0;
        to_vely[index] = 0;
    }

    // Do computation
    project_block_simple<<<dim3(WIDTH / 32, HEIGHT / 32, 1), dim3(32,32,1)>>>(flow, from_velx, from_vely, to_velx, to_vely);
    check_error(cudaPeekAtLastError());
    check_error(cudaDeviceSynchronize());


    // Load memory back from the GPU
    for (int j = 0; j < HEIGHT; j++)
    for (int i = 0; i < WIDTH; i++)
    {
        int index = i + j * WIDTH;
        to[index].vel_x = to_velx[index];
        to[index].vel_y = to_vely[index];
    }

    // check_error(cudaMemcpy(C, C_device, N*N*sizeof(float), cudaMemcpyDeviceToHost));

    check_error(cudaFree(flow));
    check_error(cudaFree(from_velx));
    check_error(cudaFree(from_vely));
    check_error(cudaFree(to_velx));
    check_error(cudaFree(to_vely));
}