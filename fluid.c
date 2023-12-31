// import libaries
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>


#include "fluid.h"
#include "fluid_cuda.cuh"

FILE * file_out;

#define PRINT_CSV 1
#define PRINT_ASCII 2
#define TRACK_USAGE 4

int flags = PRINT_CSV | TRACK_USAGE;

// PROJECT - REMOVE DIVERGENCE
// ..

void project_all_iteration(struct Tile * from, struct Tile * into)
{
    for (index_t j = 0; j < HEIGHT; j++)
    for (index_t i = 0; i < WIDTH; i++)
    {
        struct Tile * t = from + i + j * WIDTH;
        struct Tile * tu = t + ( i + 1 == WIDTH ? -i : 1 );
        struct Tile * tv = t + ( j + 1 == HEIGHT ? j * -WIDTH : WIDTH );

        float s = t->density + t->density + tu->density + tv->density;
        
        s = !s ? 0 : (1 / s);

        float d = (t->vel_x + t->vel_y - tu->vel_x - tv->vel_y) * s;

        (t - from + into)->vel_x -= d * t->density;
        (t - from + into)->vel_y -= d * t->density;
        (tu - from + into)->vel_x += d * tu->density;
        (tv - from + into)->vel_y += d * tv->density;
    }
}

// remove divergence, use jacobi
void project_all(struct Tile * board, struct Tile * board_)
{
    memcpy(board_, board, board_size);
    for (int j = 0; j < SOLVER_ITERATIONS; j++)
    {
        project_all_iteration(board, board_);
        memcpy(board, board_, board_size);
    }
}

// SIMD HELPERS
// ..

static inline float get_s_value(struct Tile * from, index_t i, index_t j)
{
    int index = i + j * WIDTH;
    float s = 2 * from[index].density  + from[index + ( i + 1 == WIDTH ? -i : 1 ) ].density + from[index + ( j + 1 == HEIGHT ? j * -WIDTH : WIDTH )].density;
    s = !s ? 0 : (1 / s);
    return s;
}


void populate_simd(struct Tile * from, struct Tile * to, __m256 * precomp_s, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely, __m256 * temp)
{
    #define vec8(tile, board, i, j, prop) _mm256_set_ps( tile( board , i + 7, j)prop, tile( board , i + 6, j)prop, tile( board , i + 5, j)prop, tile( board , i + 4, j)prop, tile( board , i + 3, j)prop, tile( board , i + 2, j)prop, tile( board , i + 1, j)prop, tile( board , i + 0, j)prop )
    
    // #pragma omp parallel for num_threads(6)
    for (index_t j = 0; j < HEIGHT + 1; j++)
    for (index_t i = 0; i < WIDTH + 1; i += 8)
    {
        int index = (i + j * WIDTH) / 8;
        // precomp_s[index] = vec8(get_s_value, from, i, j, );
        density[index] = vec8(tile_unsafe, from, i, j, .density);
        from_velx[index] = vec8(tile_unsafe, from, i, j, .vel_x);
        from_vely[index] = vec8(tile_unsafe, from, i, j, .vel_y);
        to_velx[index] = vec8(tile_unsafe, to, i, j, .vel_x);
        to_vely[index] = vec8(tile_unsafe, to, i, j, .vel_y);
        temp[index] = vec8(tile_unsafe, to, i, j, .temp);
    }
} 

void unpopulate_simd(struct Tile * from, struct Tile * to, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely, __m256 * temp)
{
    float velx[8];
    float vely[8];
    float temp_[8];
    
    for (index_t j = 0; j < HEIGHT; j++)
    for (index_t i = 0; i < WIDTH; i += 8)
    {
        int index = (i + j * WIDTH) / 8;
        _mm256_storeu_ps(velx, to_velx[index]);
        _mm256_storeu_ps(vely, to_vely[index]);
        _mm256_storeu_ps(temp_, temp[index]);
        
        for (int k = 0; k < 8; k++)
        {
            tile(to, i + k, j).vel_x = velx[k];
            tile(to, i + k, j).vel_y = vely[k];
            tile(to, i + k, j).temp = temp_[k];
        }
    }
} 

// the same as `project_all_iteration` but uses simd
static inline void project_all_fast_iteration( __m256 * precomp_s, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely, __m256 zeros, __m256 mask, __m256i permute )
{
    #define bump(place) (_mm256_loadu_ps(((float *) (place)) + 1))
    
    for (index_t i = 0; i < WIDTH * HEIGHT / 8; i++)
    {
        __m256 tu_density = bump(density + i);
        index_t tv_i = i + (WIDTH/8);

        // d = from_velx[i] + from_vely[i] - from_velx[i + 1] - from_vely[i + WIDTH * 1];
        // s = density[i] + density[i] - density[i + 1] - density[i + WIDTH * 1];
        // d *= 1 / s[i];
        __m256 d = _mm256_mul_ps(_mm256_sub_ps( _mm256_add_ps( from_velx[i], from_vely[i] ), _mm256_add_ps( bump(from_velx + i), from_vely[ tv_i ] )), precomp_s[i]);

        // to_velx[i + 1] += density[i + 1] * d;
        __m256 toadd = _mm256_permutevar8x32_ps( _mm256_mul_ps( tu_density, d ), permute );
        to_velx[i] = _mm256_add_ps(to_velx[i], _mm256_andnot_ps(mask, toadd));
        to_velx[i+1] = _mm256_add_ps(to_velx[i+1], _mm256_and_ps(mask, toadd));
        
        // to_vely[i + WIDTH * 1] += density[i + WIDTH * 1] * d;
        to_vely[tv_i] = _mm256_fmadd_ps( density[tv_i], d, to_vely[tv_i] );
        
        // to_velx[i] -= density[i] * d;
        // to_vely[i] -= density[i] * d;
        d = _mm256_mul_ps( density[i], d );
        to_velx[i] = _mm256_sub_ps( to_velx[i], d );
        to_vely[i] = _mm256_sub_ps( to_vely[i], d );
    }
}

// remove divergence from velocity, use jacobi, but fast
void project_all_fast(struct Tile * from, struct Tile * to, __m256 * precomp_s, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely, __m256 * temp)
{

    memcpy(to, from, board_size);
    populate_simd(from, to, precomp_s, density, from_velx, from_vely, to_velx, to_vely, temp);

    __m256 zeros = _mm256_set_ps(0,0,0,0,0,0,0,0);
    __m256 mask = (__m256) _mm256_set_epi32(0,0,0,0,0,0,0,-1);
    __m256i permute = _mm256_set_epi32(6,5,4,3,2,1,0,7);

    for (int r = 0; r < SOLVER_ITERATIONS; r++)
    {
        project_all_fast_iteration(precomp_s, density, from_velx, from_vely, to_velx, to_vely, zeros, mask, permute);
        memcpy(from_velx, to_velx, simd_board_size * 2);
    }

    unpopulate_simd(from, to, density, from_velx, from_vely, to_velx, to_vely, temp);
    memcpy(from, to, board_size);
}

// ADVECTION - APPLY VELOCITIES
// ..

// compute the total flow between two points
float density_between(struct Tile * board, index_t i, index_t j, index_t i_, index_t j_)
{
    if (i < -WIDTH || i_ < -WIDTH || j < -HEIGHT || j_ < -HEIGHT) return 0;

    float out = tile(board, i_, j_).density 
        * tile(board, i_, j).density 
        * tile(board, i, j_).density 
        * tile(board, i, j).density;

    if (i_ > i) for ( index_t ii = i; ii < i_; ii++ ) out *= tile(board, ii, j).density;
        else for ( index_t ii = i_; ii < i; ii++ ) out *= tile(board, ii, j).density;

    if (j_ > j) for ( index_t jj = j; jj < j_; jj++ ) out *= tile(board, i_, jj).density;
        else for ( index_t jj = j_; jj < j; jj++ ) out *= tile(board, i_, jj).density;
    
    float final_out = out;

    out = tile(board, i_, j_).density 
        * tile(board, i_, j).density 
        * tile(board, i, j_).density 
        * tile(board, i, j).density;

    if (i_ > i) for ( index_t ii = i; ii < i_; ii++ ) out *= tile(board, ii, j_).density;
        else for ( index_t ii = i_; ii < i; ii++ ) out *= tile(board, ii, j_).density;

    if (j_ > j) for ( index_t jj = j; jj < j_; jj++ ) out *= tile(board, i, jj).density;
        else for ( index_t jj = j_; jj < j; jj++ ) out *= tile(board, i, jj).density;

    return (out > final_out) ? out : final_out;
}

// apply velocity to data
void advect(struct Tile * from, struct Tile * into, index_t i, index_t j, index_t i_, index_t j_, float frac)
{
    if (i >= WIDTH - 1 || j >= HEIGHT - 1 || i_ >= WIDTH - 1 || j_ >= HEIGHT - 1 || i <= 0 || j <= 0 || i_ <= 0 || j_ <= 0 ) return;


    struct Tile * f = &(tile(from, i_, j_));
    struct Tile * x = &(tile(into, i, j));
    struct Tile * y = &(tile(into, i_, j_));

    float velx = frac * f->vel_x;
    float vely = frac * f->vel_y;
    float temp = frac * f->temp;

    float d = density_between(from, i, j, i_, j_);

    x->vel_x -= velx * d;
    x->vel_y -= vely * d;
    x->temp -= temp * d;

    y->vel_x += velx * d;
    y->vel_y += vely * d;
    y->temp += temp * d;

}

static inline void advect_tile(struct Tile * from, struct Tile * into, index_t i, index_t j, index_t i_min, index_t i_max, index_t j_min, index_t j_max)
{
    // if (tile(from, i, j).flags & ADVECTED) return;

    float vel_x = DELTA_TIME * tile(from, i, j).vel_x;
    float vel_y = DELTA_TIME * tile(from, i, j).vel_y;
    
    index_t i_ = i - vel_x;
    index_t j_ = j - vel_y;


    if ( i >= i_max || j >= j_max || i_ + 1 >= i_max || j_ + 1 >= j_max || i <= i_min || j <= j_min || i_ <= i_min || j_ <= j_min ) return;

    float frac_i = (i + vel_x) - i_;
    float frac_j = (j + vel_y) - j_;

    advect(from, into, i, j, i_,     j_,     (1 - frac_i) * (1 - frac_j));
    advect(from, into, i, j, i_ + 1, j_,     frac_i       * (1 - frac_j));
    advect(from, into, i, j, i_,     j_ + 1, (1 - frac_i) * frac_j      );
    advect(from, into, i, j, i_ + 1, j_ + 1, frac_i       * frac_j      );

    // tile(from, i, j).flags |= ADVECTED;
}

void advect_all(struct Tile * from, struct Tile * into)
{
    memcpy(into, from, board_size);

    // doing this in paralell slows down the program.
    // how can I make it speed it up instead?
    // #pragma omp parallel for
    // for (int N = 0; N < 4; N++) 
    // {
    //     int min_y = N * HEIGHT / 4;
    //     int max_y = (N + 1) * HEIGHT / 4;

    //     for (index_t y = min_y; y < max_y; y++)
    //     for (index_t x = 0; x < WIDTH; x++)
    //     {
    //         advect_tile(from, into, x, y, 0, WIDTH, min_y, max_y);
    //     }
    // }

    for (index_t y = 0; y < HEIGHT; y++)
    for (index_t x = 0; x < WIDTH; x++)
    {
        if (from[x + y * WIDTH].flags) continue;
        advect_tile(from, into, x, y, 0, WIDTH, 0, HEIGHT);
    }

    memcpy(from, into, board_size);
}

void sticky_all(struct Tile * from, struct Tile * into)
{
    // for (int c = 0; c < ; c++) 
    {
        // #pragma omp parallel for
        for (index_t y = 0; y < HEIGHT; y++)
        for (index_t x = 0; x < WIDTH; x++) 
        {
            float count = 
                ( tile(from, x + 1, y).density 
                + tile(from, x - 1, y).density 
                + tile(from, x, y + 1).density 
                + tile(from, x, y - 1).density
                );

            float next
                =
                ( tile(from, x + 1, y).temp * tile(from, x + 1, y).density
                + tile(from, x - 1, y).temp * tile(from, x - 1, y).density
                + tile(from, x, y + 1).temp * tile(from, x, y + 1).density
                + tile(from, x, y - 1).temp * tile(from, x, y - 1).density )
                / ( count ? count : 1.0 );
            
            tile(into, x, y).temp = (tile(from, x, y).temp * (1 - DELTA_TIME) + next * DELTA_TIME) * tile(from, x, y).density;
                
        }

        memcpy(from, into, board_size);
    }
}

// OTHER BITS
// ..

// apply external constraints to board
void external_consts(struct Tile * inplace, float t)
{
    // #pragma omp parallel for
    for (int i = 0; i < WIDTH * HEIGHT; i++)
    {
        inplace[i].flags = 0;

        inplace[i].vel_y += DELTA_TIME * inplace[i].temp;
        inplace[i].temp *= pow(0.98, DELTA_TIME);
        inplace[i].vel_y *= (fabs(inplace[i].vel_y) > 0.5 * HEIGHT) ? 0 : 1;
        inplace[i].vel_x *= (fabs(inplace[i].vel_x) > 0.5 * WIDTH) ? 0 : 1;

        float cap = 10;

        inplace[i].temp = (inplace[i].temp > cap) ? cap : inplace[i].temp;

        inplace[i].vel_x = (inplace[i].vel_x > cap) ? cap : inplace[i].vel_x;
        inplace[i].vel_x = (inplace[i].vel_x < -cap) ? -cap : inplace[i].vel_x;

        inplace[i].vel_y = (inplace[i].vel_y > cap) ? cap : inplace[i].vel_y;
        inplace[i].vel_y = (inplace[i].vel_y < -cap) ? -cap : inplace[i].vel_y;
    }
}

// print the board
void print_board(struct Tile * board)
{
    #define tileset(i) (" .,:;ilw8WM")[(int) ((i < 0)?0:((i*10 < 10)?i*10:10))]

    for (index_t y = HEIGHT - 1; y >= 0; y--)
    {

        for (index_t x = 0; x < WIDTH; x++) 
        {
            struct Tile t = tile(board, x, y);
            // if (t.density < 0.5) {printf("@"); continue;}
            // printf( "%c", tileset(0.5 * (fabs(t.vel_x) + fabs(t.vel_y))) );
            if (flags & PRINT_ASCII) fprintf( stderr, "%c", tileset(sqrt(t.temp + 0.001)) );
            if (flags & PRINT_CSV) fprintf(file_out, "%.2f,", t.temp);
        }
        if (flags & PRINT_CSV) fprintf(file_out, "\n");
        if (flags & PRINT_ASCII) fprintf( stderr, "\n" );
    }
}

// init board
void init_board(struct Tile * board)
{
    for (int i = 0; i < WIDTH; i++)
    for (int j = 0; j < HEIGHT; j++)
    {
        tile(board, i, j).flags = 0;
        tile(board, i, j).vel_x = 0.0;
        tile(board, i, j).vel_y = 0.0;
        // tile(board, i, j).temp = (rand() % 10)*0.01 - 0.05;
        tile(board, i, j).density = ! ( !i || !j || i + 1 >= WIDTH || j + 1 >= HEIGHT );
        // tile(board, i, j).density *= ! ( pow(i - WIDTH/2, 2) + pow((j * 2) - 2 * 0.7 * HEIGHT, 2) < pow(7, 2) );
        tile(board, i, j).density *= ((int) (fabs(i - WIDTH / 2) >= 15) || (fabs(j - 2 * HEIGHT / 3) >= 5));
    }
}


clock_t measure_clock_prev = 0;

float measure()
{
    float out = (float) (clock() - measure_clock_prev) / CLOCKS_PER_SEC;
    measure_clock_prev = clock();
    return out;
}

int main(int argc, char** argv) 
{
    // ALLOCATE GRID DATA
    struct Tile * board = (struct Tile *) malloc(board_size);
    struct Tile * board_ = (struct Tile *) malloc(board_size);

    // ALLOCATE BUFFER FOR SIMD FUNCTION USE
    __m256 * precomp_s = (__m256 *) aligned_alloc(32, simd_board_size);
    __m256 * density = (__m256 *) aligned_alloc(32, simd_board_size);
    __m256 * from_velx = (__m256 *) aligned_alloc(32, simd_board_size * 2);
    __m256 * from_vely = from_velx + simd_board_size / sizeof(__m256);
    __m256 * to_velx = (__m256 *) aligned_alloc(32, simd_board_size * 2);
    __m256 * to_vely = to_velx + simd_board_size / sizeof(__m256);
    __m256 * from_temp = (__m256 *) aligned_alloc(32, simd_board_size);
    __m256 * to_temp = (__m256 *) aligned_alloc(32, simd_board_size);

    float * cuda_density = cuda_alloc();
    float * cuda_from_velx = cuda_alloc();
    float * cuda_from_vely = cuda_alloc();
    float * cuda_to_velx = cuda_alloc();
    float * cuda_to_vely = cuda_alloc();
    float * cuda_pressure = cuda_alloc();
    float * cuda_precomp_s = cuda_alloc();
    float * cuda_from_temp = cuda_alloc();
    float * cuda_to_temp = cuda_alloc();

    // INITIALISE GRID DATA
    init_board(board);
    memcpy(board_, board, board_size);

    if (flags & PRINT_CSV) {
        file_out = fopen("./out.csv","w");
        if (!file_out) 
        {
            fprintf(stderr, "file open failed :(\n");
            return -1;
        }
    }
    if (flags & PRINT_CSV) fprintf(file_out, "%d,%d\n", WIDTH, HEIGHT);

    float external_time = 0;
    float project_time = 0;
    float advect_time = 0;
    float other_time = 0;

    measure();
    fprintf(stderr, "RENDERING %d FRAMES:", FRAMES);

    // ENTER MAIN LOOP

    for (int i = 0; i < FRAMES; i++)
    {
        print_board(board);
        fprintf(stderr, ".");

        for (int k = 0; k < TIME_PER_FRAME / DELTA_TIME; k++)
        {
            if (flags & TRACK_USAGE) other_time += measure();

            // external_consts(board, (float) i + k * DELTA_TIME);

            tile(board, WIDTH/2, HEIGHT/4).temp = 2;
            tile(board, WIDTH/2, HEIGHT/4).vel_x += DELTA_TIME * sin(((float) i + k * DELTA_TIME) * 0.25) * 8;

            if (flags & TRACK_USAGE) external_time += measure();

            // project_all(board, board_);
            // project_all_fast(board, board_, precomp_s, density, from_velx, from_vely, to_velx, to_vely, from_temp);
            project_all_gpu(board, board_, precomp_s, density, from_velx, from_vely, to_velx, to_vely, from_temp, to_temp, cuda_density, cuda_from_velx, cuda_from_vely, cuda_to_velx, cuda_to_vely, cuda_pressure, cuda_precomp_s, cuda_from_temp, cuda_to_temp);
            if (flags & TRACK_USAGE) project_time += measure();

            advect_all(board, board_);

            if (flags & TRACK_USAGE) advect_time += measure();
            // sticky_all(board, board_);
        }
    }

    if (flags & TRACK_USAGE) {
        float total = external_time + project_time + advect_time + other_time;
        fprintf(stderr, "\n-- USAGE REPORT (avg fps. %f) --\n", FRAMES / total );
        fprintf(stderr, "\n CONSTRAINTS \t %f%%\n", (external_time / total) * 100 );
        fprintf(stderr, "\n PROJECTION  \t %f%%\n", (project_time / total) * 100 );
        fprintf(stderr, "\n ADVECTION   \t %f%%\n", (advect_time / total) * 100 );
        fprintf(stderr, "\n OTHER       \t %f%%\n", (other_time / total) * 100 );
        fprintf(stderr, "\n NOTES       \t \"%s\"\n", NOTES );
    }

    return 0;
}