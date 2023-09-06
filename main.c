#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

#define WIDTH 128
#define HEIGHT 32

#define tile(board, x, y) (board)[ ((int) ((y + HEIGHT) % HEIGHT) * WIDTH) + ((x + WIDTH) % WIDTH) ]

#define index_t int16_t
#define board_size (sizeof(struct Tile) * WIDTH * HEIGHT)

#define delta_time 0.05

#define tileset(i) (" .,:;ilw8WM")[(int) ((i < 0)?0:((i*10 < 10)?i*10:10))]

struct Tile 
{
    double vel_x;
    double vel_y;
    double density;
    double temp;
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


// void jacobi(half2 coords   : WPOS,   // grid coordinates

//    out
//    half4 xNew : COLOR,  // result

//    uniform half alpha // -pressure^2
//    , uniform half rBeta // 0.25
//    , uniform samplerRECT x // pressure
//    , uniform samplerRECT b // divergence
//    )   // b vector (Ax = b)
// {
//     vec l = tile(pressure, i - 1, j);
//     vec b = tile(pressure, i, j - 1);
//     vec r = tile(pressure, i + 1, j);
//     vec t = tile(pressure, i, j + 1);

//     vec div = tile(divergence, i, j);

//     // evaluate Jacobi iteration
//     tile(pressure, i, j) = (l + r + b + t - (pressure)^2 * div) * 0.25;
// }


// // solve A * pressure = divergence
// void jacobi_div(struct Tile * from, struct Tile * into, double alpha, double rbeta)
// {
//     for (index_t j = 0; j < HEIGHT; j++)
//     for (index_t i = 0; i < WIDTH; i++)
//     {
//         struct Tile * t = from + i + j * WIDTH;
//         struct Tile * tu = t + ( i + 1 == WIDTH ? -i : 1 );
//         struct Tile * tv = t + ( j + 1 == HEIGHT ? j * -WIDTH : WIDTH );
//         struct Tile * tx = t + ( !i ? i + WIDTH - 1 : -1 );
//         struct Tile * ty = t + ( !j ? WIDTH * (HEIGHT - 1) : -WIDTH );

//         double div = (t->vel_x - tu->vel_x + t->vel_y - tv->vel_y); 

//         double w = tx->density + ty->density + tu->density + tv->density;

//         (t - from + into)->p_x = 
//             ( tx->p_x * tx->density
//             + ty->p_x * ty->density
//             + tu->p_x * tu->density
//             + tv->p_x * tv->density
//             + alpha * div
//             ) * (w == 0? 1 : 1.0 / w );
//             // + - t->density ;
//     }
// }


// void jacobi(struct Tile * from, struct Tile * into)
// {
//     for (index_t j = 0; j < HEIGHT; j++)
//     for (index_t i = 0; i < WIDTH; i++)
//     {
//         struct Tile * t = from + i + j * WIDTH;
//         struct Tile * tu = t + ( i + 1 == WIDTH ? -i : 1 );
//         struct Tile * tv = t + ( j + 1 == HEIGHT ? j * -WIDTH : WIDTH );
//         // struct Tile * tx = t + ( !i ? i + WIDTH - 1 : -1 );
//         // struct Tile * ty = t + ( !j ? WIDTH * (HEIGHT - 1) : -WIDTH );

//         double dens = t->density + t->density + tu->density + tv->density;
        
//         dens = (fabs(dens) < 0.01) ? 0 : (1.0 / dens);

//         double d = (t->vel_x + t->vel_y - tu->vel_x - tv->vel_y) * dens;

//         (t - from + into)->p_x -= d * t->density;
//         (t - from + into)->p_y -= d * t->density;
//         (tu - from + into)->p_x += d * tu->density;
//         (tv - from + into)->p_y += d * tv->density;

//         // (t - from + into)->p_x = t->density *
//         //     ( t->p_x * t->density + (1 - t->density)
//         //     + t->p_y * t->density + (1 - t->density)
//         //     + tu->p_x * tu->density + (1 - tu->density)
//         //     + tv->p_y * tv->density + (1 - tv->density)
//         //     - pressure2 * (t->vel_x + t->vel_y - tu->vel_x - tv->vel_y)
//         //     ) * 0.25;
//     }
// }

// void applygrad(struct Tile * board)
// {
//     for (index_t j = 0; j < HEIGHT; j++)
//     for (index_t i = 0; i < WIDTH; i++)
//     {
//         struct Tile * t = board + i + j * WIDTH;
//         // struct Tile * tu = t + ( i + 1 == WIDTH ? -i : 1 );
//         // struct Tile * tv = t + ( j + 1 == HEIGHT ? j * -WIDTH : WIDTH );
//         struct Tile * tx = t + ( !i ? i + WIDTH - 1 : -1 );
//         struct Tile * ty = t + ( !j ? WIDTH * (HEIGHT - 1) : -WIDTH );

//         t->vel_x += delta_time * t->density * tx->p_x;
//         t->vel_y += delta_time * t->density * ty->p_x;
//         t->vel_x -= delta_time * t->density * t->p_x;
//         t->vel_y -= delta_time * t->density * t->p_x;
//     }
// }

void project_all(struct Tile * from, struct Tile * into)
{
    for (index_t j = 0; j < HEIGHT; j++)
    for (index_t i = 0; i < WIDTH; i++)
    {
        struct Tile * t = from + i + j * WIDTH;
        struct Tile * tu = t + ( i + 1 == WIDTH ? -i : 1 );
        struct Tile * tv = t + ( j + 1 == HEIGHT ? j * -WIDTH : WIDTH );

        double s = t->density + t->density + tu->density + tv->density;
        
        s = (fabs(s) < 0.01) ? 0 : (1 / s);

        double d = (t->vel_x + t->vel_y - tu->vel_x - tv->vel_y) * s;
        
        (t - from + into)->vel_x -= d * t->density;
        (t - from + into)->vel_y -= d * t->density;
        (tu - from + into)->vel_x += d * tu->density;
        (tv - from + into)->vel_y += d * tv->density;
    }
}


double density_between(struct Tile * board, index_t i, index_t j, index_t i_, index_t j_)
{
    if (i < -WIDTH || i_ < -WIDTH || j < -HEIGHT || j_ < -HEIGHT) return 0;

    // printf("(%d, %d) (%d, %d)\n", i, j, i_, j_);
    double out = tile(board, i_, j_).density;

    if (i_ > i) for ( index_t ii = i; ii < i_; ii++ ) out *= tile(board, ii, j).density;
        else    for ( index_t ii = i_; ii < i; ii++ ) out *= tile(board, ii, j).density;

    if (j_ > j) for ( index_t jj = j; jj < j_; jj++ ) out *= tile(board, i_, jj).density;
        else    for ( index_t jj = j_; jj < j; jj++ ) out *= tile(board, i_, jj).density;

    return out;
}

void advect(struct Tile * from, struct Tile * into, index_t i, index_t j, index_t i_, index_t j_, double frac)
{

    struct Tile * f = &(tile(from, i_, j_));
    struct Tile * x = &(tile(into, i, j));
    struct Tile * y = &(tile(into, i_, j_));

    double velx = frac * f->vel_x;
    double vely = frac * f->vel_y;
    double temp = frac * f->temp;

    double d = density_between(from, i, j, i_, j_);

    x->vel_x -= velx;
    x->vel_y -= vely;
    x->temp -= temp;

    y->vel_x += velx;
    y->vel_y += vely;
    y->temp += temp;
}

static inline void advect_tile(struct Tile * from, struct Tile * into, index_t i, index_t j)
{
    double vel_x = delta_time * tile(from, i, j).vel_x;
    double vel_y = delta_time * tile(from, i, j).vel_y;
    
    index_t i_ = i - vel_x;
    index_t j_ = j - vel_y;

    double frac_i = (i + vel_x) - i_;
    double frac_j = (j + vel_y) - j_;

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
        // inplace[i].p_x = 1;
        // inplace[i].p_y = 1;
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

    for (int i = 0; i < WIDTH; i++) 
    for (int j = 0; j < HEIGHT; j++) 
    {
        tile(board, i, j).density = ! ( !i || !j || i + 1 >= WIDTH || j + 1 >= HEIGHT );
        tile(board, i, j).density *= ! ( pow(i - WIDTH/2, 2) + pow((j * 2) - 2 * 0.7 * HEIGHT, 2) < pow(7, 2) );
        
        tile(board, i, j).vel_x = 0.0;
        tile(board, i, j).vel_y = 0.0;
        tile(board, i, j).temp = 0.0;
        // tile(board, i, j).p_x = 0.0;
        // tile(board, i, j).p_y = 0.0;
    }

    memcpy(board_, board, board_size);

    for (int i = 0; i < 1000; i++)
    {
        print_board(board);

        for (int k = 0; k < 1 / delta_time; k++) 
        {
            external_consts(board);
            memcpy(board_, board, board_size);
            for (int j = 0; j < 40; j++)
            {
                
                project_all(board, board_);
                memcpy(board, board_, board_size);
                // project_all(board_, board);
                // jacobi_div(board, board_, -0.1, 0.25);
            }

            // applygrad(board);
    
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