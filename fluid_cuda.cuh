void project_all_gpu(
    struct Tile * from, struct Tile * to, 
    __m256 * ext_s, __m256 * ext_flow, __m256 * ext_from_velx, __m256 * ext_from_vely, __m256 * ext_to_velx, __m256 * ext_to_vely, 
    float * flow, float * from_velx, float * from_vely, float * to_velx, float * to_vely, float * pressure, float * cuda_precomp_s );

float * cuda_alloc();