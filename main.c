#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

#define WIDTH 128
#define HEIGHT 32
#define ITCOUNT 60

#define tile(board, x, y) (board)[ ((int) ((y + HEIGHT) % HEIGHT) * WIDTH) + ((x + WIDTH) % WIDTH) ]

#define index_t int16_t
#define board_size (sizeof(struct Tile) * WIDTH * HEIGHT)
#define simd_board_size (sizeof(__m256) * (WIDTH/8 + 1) * (HEIGHT + 1))

#define delta_time 0.05

#define tileset(i) (" .,:;ilw8WM")[(int) ((i < 0)?0:((i*10 < 10)?i*10:10))]

struct Tile 
{
    float temp;
    float vel_x;
    float vel_y;
    float density;
};


/*

    +-v-+---+---+---
    x   u   |   |
    +-y-+---+---+---
    |   |   |   |
    +---+---+---+---
    |   |   |   |
   0,0--+---+---+---

*/

void project_all(struct Tile * from, struct Tile * into)
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
        
        // 16.75 s
        (t - from + into)->vel_x -= d * t->density;
        (t - from + into)->vel_y -= d * t->density;
        (tu - from + into)->vel_x += d * tu->density;
        (tv - from + into)->vel_y += d * tv->density;
    }

    // for (index_t i = 0; i < WIDTH*HEIGHT; i++) 19.09 s
    // {
    //     into[i].vel_x *= from[i].density;
    //     into[i].vel_y *= from[i].density;
    // }

}



/*
    17.63 s
*/
void do_project_all(struct Tile * board, struct Tile * board_)
{
    memcpy(board_, board, board_size);
    for (int j = 0; j < ITCOUNT; j++)
    {
        project_all(board, board_);
        memcpy(board, board_, board_size);
    }
}

#define bump(place) (_mm256_loadu_ps(((float *) (&place)) + 1))

void project_single_fast( __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely, index_t i )
{
    __m256 zeros = _mm256_set_ps(0,0,0,0,0,0,0,0);
    __m256 mask = (__m256) _mm256_set_epi32(0,0,0,0,0,0,0,-1);
    __m256i permute = _mm256_set_epi32(6,5,4,3,2,1,0,7);
    
    __m256 tu_density = bump(density[i]);
    index_t tv_i = i + WIDTH/8;


    /*
        d = from_velx[i] + from_vely[i] - from_velx[i + 0.1] - from_vely[i + WIDTH * 0.1];
    */
    __m256 d = _mm256_sub_ps( _mm256_add_ps( from_velx[i], from_vely[i] )
                            , _mm256_add_ps( bump(from_velx[i]), from_vely[ tv_i ] )
                            );

    
    __m256 s = _mm256_add_ps( _mm256_add_ps( density[i], density[i])
                            , _mm256_add_ps( tu_density, density[ tv_i ] )
                            );


    s = _mm256_and_ps(_mm256_cmp_ps(s, zeros, _CMP_NEQ_OS), _mm256_rcp_ps(s));
    /*
        d *= s[i];
    */
    d = _mm256_mul_ps( d, s );
    
    /*
        to_velx[i + 1] += density[i + 1] * d;
    */

    __m256 toadd = _mm256_permutevar8x32_ps( _mm256_add_ps( bump(to_velx[i]), _mm256_mul_ps(tu_density, d) ), permute );
    to_velx[i] = _mm256_add_ps(_mm256_and_ps(mask, to_velx[i]), _mm256_andnot_ps(mask, toadd));
    to_velx[i+1] = _mm256_add_ps(_mm256_andnot_ps(mask, to_velx[i+1]), _mm256_and_ps(mask, toadd));
    
    
    /* 
        to_vely[i + WIDTH * 0.1] += density[i + WIDTH * 0.1] * d;
    */
    to_vely[tv_i] = _mm256_add_ps( to_vely[tv_i], _mm256_mul_ps(density[tv_i], d) );
    
    /*
        to_velx[i] -= density[i] * d;
        to_vely[i] -= density[i] * d;
    */
    d = _mm256_mul_ps( density[i], d );

    to_velx[i] = _mm256_sub_ps( to_velx[i], d );
    to_vely[i] = _mm256_sub_ps( to_vely[i], d );
}


static inline void populate_simd(struct Tile * from, struct Tile * to, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely)
{
    for (index_t j = 0; j < HEIGHT + 1; j++)
    for (index_t i = 0; i < WIDTH + 1; i += 8)
    {
        int index = (i + j * WIDTH) / 8;
        
        #define vec8(board, i, j, prop) _mm256_set_ps( tile( board , i + 7, j).prop, tile( board , i + 6, j).prop, tile( board , i + 5, j).prop, tile( board , i + 4, j).prop, tile( board , i + 3, j).prop, tile( board , i + 2, j).prop, tile( board , i + 1, j).prop, tile( board , i + 0, j).prop )
        
        density[index] = vec8(from, i, j, density);
        from_velx[index] = vec8(from, i, j, vel_x);
        from_vely[index] = vec8(from, i, j, vel_y);
        to_velx[index] = vec8(to, i, j, vel_x);
        to_vely[index] = vec8(to, i, j, vel_y);
    }
} 

static inline void unpopulate_simd(struct Tile * from, struct Tile * to, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely)
{
    for (index_t j = 0; j < HEIGHT; j++)
    for (index_t i = 0; i < WIDTH; i += 8)
    {
        float velx[8];
        float vely[8];
        int index = (i + j * WIDTH) / 8;
        _mm256_storeu_ps(velx, ((__m256 *) to_velx)[index]);
        _mm256_storeu_ps(vely, ((__m256 *) to_vely)[index]);
        
        for (int k = 0; k < 8; k++)
        {
            tile(to, i + k, j).vel_x = velx[k];
            tile(to, i + k, j).vel_y = vely[k];
        }
    }
} 

/*
    8.23 s
*/
void project_all_fast(struct Tile * from, struct Tile * to, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely)
{
    memcpy(to, from, board_size);

    populate_simd(from, to, density, from_velx, from_vely, to_velx, to_vely);

    int c = WIDTH * HEIGHT / 8;

    for (int r = 0; r < ITCOUNT; r++)
    {
        for (index_t i = 0; i < c; i++)
            project_single_fast(density, from_velx, from_vely, to_velx, to_vely, i);

        memcpy(from_velx, to_velx, simd_board_size); //_mm256_mul_ps(to_velx[i], density[i]);
        memcpy(from_vely, to_vely, simd_board_size); //_mm256_mul_ps(to_vely[i], density[i]);
    }

    unpopulate_simd(from, to, density, from_velx, from_vely, to_velx, to_vely);

    memcpy(from, to, board_size);
}



float density_between(struct Tile * board, index_t i, index_t j, index_t i_, index_t j_)
{
    if (i < -WIDTH || i_ < -WIDTH || j < -HEIGHT || j_ < -HEIGHT) return 0;

    float out = tile(board, i_, j_).density;

    if (i_ > i) for ( index_t ii = i; ii < i_; ii++ ) out *= tile(board, ii, j).density;
        else    for ( index_t ii = i_; ii < i; ii++ ) out *= tile(board, ii, j).density;

    if (j_ > j) for ( index_t jj = j; jj < j_; jj++ ) out *= tile(board, i_, jj).density;
        else    for ( index_t jj = j_; jj < j; jj++ ) out *= tile(board, i_, jj).density;

    return out;
}

void advect(struct Tile * from, struct Tile * into, index_t i, index_t j, index_t i_, index_t j_, float frac)
{

    struct Tile * f = &(tile(from, i_, j_));
    struct Tile * x = &(tile(into, i, j));
    struct Tile * y = &(tile(into, i_, j_));

    float velx = frac * f->vel_x;
    float vely = frac * f->vel_y;
    float temp = frac * f->temp;

    float d = density_between(from, i, j, i_, j_);

    x->vel_x -= velx;
    x->vel_y -= vely;
    x->temp -= temp;

    y->vel_x += velx;
    y->vel_y += vely;
    y->temp += temp;
}

static inline void advect_tile(struct Tile * from, struct Tile * into, index_t i, index_t j)
{
    float vel_x = delta_time * tile(from, i, j).vel_x;
    float vel_y = delta_time * tile(from, i, j).vel_y;
    
    index_t i_ = i - vel_x;
    index_t j_ = j - vel_y;

    float frac_i = (i + vel_x) - i_;
    float frac_j = (j + vel_y) - j_;

    advect(from, into, i, j, i_,     j_,     (1 - frac_i) * (1 - frac_j));
    advect(from, into, i, j, i_ + 1, j_,     frac_i       * (1 - frac_j));
    advect(from, into, i, j, i_,     j_ + 1, (1 - frac_i) * frac_j      );
    advect(from, into, i, j, i_ + 1, j_ + 1, frac_i       * frac_j      );
}

void advect_all(struct Tile * from, struct Tile * into)
{
    for (index_t y = 0; y < HEIGHT; y++)
    for (index_t x = 0; x < WIDTH; x++) 
    {
        advect_tile(from, into, x, y);
    }
}

void external_consts(struct Tile * inplace) 
{
    for (int i = 0; i < WIDTH * HEIGHT; i++) 
    {
        inplace[i].vel_y += delta_time * inplace[i].temp;
        inplace[i].temp *= pow(0.99, delta_time);
        inplace[i].vel_y *= (fabs(inplace[i].vel_y) > 0.5 * HEIGHT) ? 0 : 1;
        inplace[i].vel_x *= (fabs(inplace[i].vel_x) > 0.5 * WIDTH) ? 0 : 1;
    }

    tile(inplace, WIDTH/2, HEIGHT/4).temp = 1;
}

void print_board(struct Tile * board)
{
    printf("\n\n FLUID SIM \n\n\n");

    for (index_t y = HEIGHT - 1; y >= 0 ; y--)
    {
        for (index_t x = 0; x < WIDTH; x++) 
        {
            struct Tile t = tile(board, x, y);

            // if (t.density < 0.5) {printf("@"); continue;}
            printf( "%c", tileset(sqrt(t.temp + 0.001)) );
            // printf( "%c", tileset(0.5 * (fabs(t.p_x) + fabs(t.p_y))));
            // printf( "%c", tileset(0.5 * (fabs(t.vel_x) + fabs(t.vel_y))) );

            // if ((fabs(t.vel_y) < 0.05) && (fabs(t.vel_x) < 0.05)) printf(" ");
            // else if (fabs(t.vel_y) < 0.05) printf("-");
            // else if (fabs(t.vel_x) < 0.05) printf("|");
            // else if (t.vel_x * t.vel_y < 0) printf("/");
            // else printf("\\");
        }
        printf("\n");
    }
}

int main2() 
{
    struct Tile * board = malloc(board_size);
    struct Tile * board_ = malloc(board_size);

    __m256 * density = aligned_alloc(32, simd_board_size);
    __m256 * from_velx = aligned_alloc(32, simd_board_size);
    __m256 * from_vely = aligned_alloc(32, simd_board_size);
    __m256 * to_velx = aligned_alloc(32, simd_board_size);
    __m256 * to_vely = aligned_alloc(32, simd_board_size);

    for (int i = 0; i < WIDTH; i++) 
    for (int j = 0; j < HEIGHT; j++) 
    {
        tile(board, i, j).vel_x = 0.0;
        tile(board, i, j).vel_y = 0.0;
        tile(board, i, j).temp = 0.0;
        tile(board, i, j).density = ! ( !i || !j || i + 1 >= WIDTH || j + 1 >= HEIGHT );
        tile(board, i, j).density *= ! ( pow(i - WIDTH/2, 2) + pow((j * 2) - 2 * 0.7 * HEIGHT, 2) < pow(7, 2) );
    }

    memcpy(board_, board, board_size);

    for (int i = 0; i < 100; i++)
    {
        print_board(board);

        for (int k = 0; k < 1 / delta_time; k++) 
        {
            external_consts(board);

            project_all_fast(board, board_, density, from_velx, from_vely, to_velx, to_vely);
            // do_project_all(board, board_);
    
            memcpy(board_, board, board_size);
            advect_all(board, board_);
            memcpy(board, board_, board_size);
        }
    }
    return 0;
}

int main()
{
    for (int i = 0; i < 10; i++) main2();
}

/* TEST CODE

    __m256 mask = (__m256) _mm256_set_epi32(0,0,0,0,0,0,0,-1);
    __m256i permute = _mm256_set_epi32(6,5,4,3,2,1,0,7);
    __m256 * test = aligned_alloc(32, sizeof(__m256) * 2);
    test[0] = _mm256_set_ps(8,7,6,5,4,3,2,1);
    test[1] = _mm256_set_ps(16,15,14,13,12,11,10,9);
    __m256 test2 = bump(test[0]);
    __m256 test3 = _mm256_permutevar8x32_ps(test[0], permute);
    __m256 test4 = _mm256_and_ps(test[0], mask);

    float velx[8];
    _mm256_storeu_ps(velx, test[0]);

*/