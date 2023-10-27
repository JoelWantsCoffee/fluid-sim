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
__device__ __inline__ float get_pressure(int i, int j, float * pressure) 
{
    if ((i < 0) || (j < 0)) return 0;
    return pressure[i + j * WIDTH];
}

__global__ void compute_pressure_map(float * flow, float * pressure, float * vel_x, float * vel_y) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= WIDTH || j >= HEIGHT) return;
    
    int t = i + j * WIDTH;

    float flow_tu  = (i + 1 >= WIDTH ? flow[t] : flow[t+1]);
    float flow_tv  = (j + 1 >= HEIGHT ? flow[t] : flow[t + WIDTH]);
    float vel_x_tu = (i + 1 >= WIDTH ? vel_x[t] : vel_x[t+1]);
    float vel_y_tv = (j + 1 >= HEIGHT ? vel_y[t] : vel_y[t + WIDTH]);

    float s = 2 * flow[t] + flow_tu + flow_tv;

    pressure[t] = (s == 0) ? 0 : (vel_x[t] + vel_y[t] - vel_x_tu - vel_y_tv) / s;
}

__global__ void project_block_simple(float * flow, float * pressure, float * to_velx, float * to_vely)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= WIDTH || j >= HEIGHT) return;

    int t = i + j * WIDTH;

    to_velx[t] += (get_pressure(i - 1, j, pressure) - get_pressure(i, j, pressure)) * flow[t];
    to_vely[t] += (get_pressure(i, j - 1, pressure) - get_pressure(i, j, pressure)) * flow[t];
}

__host__ void project_all_main_loop(float * flow, float * from_velx, float * from_vely, float * to_velx, float * to_vely, float * pressure)
{
    dim3 blocks(WIDTH / 32, HEIGHT / 32, 1);  // Changed to handle extra boundary block
    dim3 threads(32, 32, 1);

    for (int i = 0; i < SOLVER_ITERATIONS; i++) 
    {
        compute_pressure_map<<<blocks, threads>>>(flow, pressure, from_velx, from_vely);
        check_error(cudaPeekAtLastError());
        check_error(cudaDeviceSynchronize());

        project_block_simple<<<blocks, threads>>>(flow, pressure, to_velx, to_vely);
        check_error(cudaPeekAtLastError());
        check_error(cudaDeviceSynchronize());

        check_error(cudaMemcpy(from_velx, to_velx, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToDevice));
        check_error(cudaMemcpy(from_vely, to_vely, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

__host__ void project_all_gpu(struct Tile * from, struct Tile * to, __m256 * ext_s, __m256 * ext_flow, __m256 * ext_from_velx, __m256 * ext_from_vely, __m256 * ext_to_velx, __m256 * ext_to_vely)
{
    // Load memory onto the GPU

    memcpy(to, from, board_size);
    populate_simd(from, to, ext_s, ext_flow, ext_from_velx, ext_from_vely, ext_to_velx, ext_to_vely);

    float * flow;    check_error(cudaMalloc(&flow, WIDTH*HEIGHT*sizeof(float)));
    float * from_velx;  check_error(cudaMalloc(&from_velx, WIDTH*HEIGHT*sizeof(float)));
    float * from_vely;  check_error(cudaMalloc(&from_vely, WIDTH*HEIGHT*sizeof(float)));
    float * to_velx;    check_error(cudaMalloc(&to_velx, WIDTH*HEIGHT*sizeof(float)));
    float * to_vely;    check_error(cudaMalloc(&to_vely, WIDTH*HEIGHT*sizeof(float)));

    check_error(cudaMemcpy(flow, ext_flow, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(from_velx, ext_from_velx, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(from_vely, ext_from_vely, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(to_velx, ext_to_velx, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(to_vely, ext_to_vely, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));

    // Do computation

    float * pressure;    
    check_error(cudaMalloc(&pressure, WIDTH*HEIGHT*sizeof(float)));
    project_all_main_loop(flow, from_velx, from_vely, to_velx, to_vely, pressure);

    // Load memory back from the GPU

    check_error(cudaMemcpy(ext_flow, flow, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));
    check_error(cudaMemcpy(ext_from_velx, from_velx, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));
    check_error(cudaMemcpy(ext_from_vely, from_vely, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));
    check_error(cudaMemcpy(ext_to_velx, to_velx, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));
    check_error(cudaMemcpy(ext_to_vely, to_vely, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));

    check_error(cudaFree(flow));
    check_error(cudaFree(from_velx));
    check_error(cudaFree(from_vely));
    check_error(cudaFree(to_velx));
    check_error(cudaFree(to_vely));

    unpopulate_simd(from, to, ext_flow, ext_from_velx, ext_from_vely, ext_to_velx, ext_to_vely);
    memcpy(from, to, board_size);
}


__device__ float get_pressure_meme(int i, int j, float * flow, float * vel_x, float * vel_y) 
{
    if ((i < 0) || (j < 0)) return 0;

    int t = i + j * WIDTH;

    int tu = t + (i + 1 >= WIDTH ? -i : 1);
    int tv = t + (j + 1 >= HEIGHT ? -j * WIDTH : WIDTH);

    float s = flow[t] + flow[t] + flow[tu] + flow[tv];
    s = (s == 0) ? 0 : 1 / s;

    return (vel_x[t] + vel_y[t] - vel_x[tu] - vel_y[tv]) * s;
}