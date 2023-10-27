#include <stdio.h>
#include <vector_types.h>
#include <immintrin.h>
#include <omp.h>

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

__host__ float * cuda_alloc()
{
    float * out;
    check_error(cudaMalloc(&out, WIDTH*HEIGHT*sizeof(float)));
    return out;
}

__global__ void compute_pressure_map(float * precomp_s, float * pressure, float * vel_x, float * vel_y) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int t = i + j * WIDTH;

    if (t >= WIDTH * HEIGHT - WIDTH) return;

    pressure[t] = precomp_s[t] * (vel_x[t] + vel_y[t] - vel_x[t+1] - vel_y[t + WIDTH]);
}

__global__ void project_block_simple(float * flow, float * pressure, float * from_velx, float * from_vely, float * to_velx, float * to_vely)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int t = i + j * WIDTH;

    if (t < WIDTH) return;

    to_velx[t] = from_velx[t] + (pressure[t - 1] - pressure[t]) * flow[t];
    to_vely[t] = from_vely[t] + (pressure[t - WIDTH] - pressure[t]) * flow[t];
}

__global__ void compute_s(float * precomp_s, float * flow, float * pressure) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int t = i + j * WIDTH;

    float flow_tu  = (i + 1 >= WIDTH ? flow[t] : flow[t+1]);
    float flow_tv  = (j + 1 >= HEIGHT ? flow[t] : flow[t + WIDTH]);

    float s = 2 * flow[t] + flow_tu + flow_tv;

    precomp_s[t] = (s == 0) ? 0 : (1/s);
    pressure[t] = 0;
}

__host__ void project_all_main_loop(float * flow, float * from_velx, float * from_vely, float * to_velx, float * to_vely, float * pressure, float * precomp_s)
{
    dim3 blocks(WIDTH / 32, HEIGHT / 32, 1);  // This can be dynamically determined based on device properties
    dim3 threads(32, 32, 1);

    compute_s<<<blocks, threads>>>(precomp_s, flow, pressure);

    for (int i = 0; i < SOLVER_ITERATIONS/2; i++) 
    {
        compute_pressure_map<<<blocks, threads>>>(precomp_s, pressure, from_velx, from_vely);
        project_block_simple<<<blocks, threads>>>(flow, pressure, from_velx, from_vely, to_velx, to_vely);
        compute_pressure_map<<<blocks, threads>>>(precomp_s, pressure, to_velx, to_vely);
        project_block_simple<<<blocks, threads>>>(flow, pressure, to_velx, to_vely, from_velx, from_vely);
    }
    check_error(cudaPeekAtLastError());
    check_error(cudaDeviceSynchronize());
}

__host__ void project_all_gpu(
    struct Tile * from, struct Tile * to, 
    __m256 * ext_s, __m256 * ext_flow, __m256 * ext_from_velx, __m256 * ext_from_vely, __m256 * ext_to_velx, __m256 * ext_to_vely, 
    float * flow, float * from_velx, float * from_vely, float * to_velx, float * to_vely, float * pressure, float * precomp_s
    )
{
    // Assume flow, from_velx, from_vely, to_velx, to_vely, and pressure are pre-allocated and passed as arguments

    // Load memory onto the GPU
    memcpy(to, from, board_size);
    populate_simd(from, to, ext_s, ext_flow, ext_from_velx, ext_from_vely, ext_to_velx, ext_to_vely);

    check_error(cudaMemcpy(flow, ext_flow, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(from_velx, ext_from_velx, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(from_vely, ext_from_vely, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(to_velx, ext_to_velx, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(to_vely, ext_to_vely, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));

    // Do computation
    project_all_main_loop(flow, from_velx, from_vely, to_velx, to_vely, pressure, precomp_s);

    // Load memory back from the GPU
    check_error(cudaMemcpy(ext_flow, flow, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));
    check_error(cudaMemcpy(ext_from_velx, from_velx, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));
    check_error(cudaMemcpy(ext_from_vely, from_vely, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));
    check_error(cudaMemcpy(ext_to_velx, to_velx, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));
    check_error(cudaMemcpy(ext_to_vely, to_vely, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));

    unpopulate_simd(from, to, ext_flow, ext_from_velx, ext_from_vely, ext_to_velx, ext_to_vely);
    memcpy(from, to, board_size);
}