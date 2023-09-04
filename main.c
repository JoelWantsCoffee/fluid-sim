#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define WIDTH 128
#define HEIGHT 32

#define tile(board, x, y) (board)[ ((int) ((y + HEIGHT) % HEIGHT) * WIDTH) + ((x + WIDTH) % WIDTH) ]

#define index_t int
#define board_size (sizeof(struct Tile) * WIDTH * HEIGHT)

#define delta_time 0.05

#define tileset(i) (" .,:;ilw8WM")[(int) ((i < 0)?0:((i*10 < 10)?i*10:10))]

#define min(x,y) (x < y ? x : y)
#define max(x,y) (x > y ? x : y)

#define sign(x) (x < 0 ? -1 : 1)

struct Tile 
{
    double vel_x;
    double vel_y;
    double density;
    double temp;
    double pressure;
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



static inline void gs_project_tile(struct Tile * from, struct Tile * into, index_t i, index_t j)
{ 
    struct Tile * t = from + i + j * WIDTH;
    struct Tile * tu = t + ( i + 1 == WIDTH ? -i : 1 );
    struct Tile * tv = t + ( j + 1 == HEIGHT ? j * -WIDTH : WIDTH );

    double s = t->density + t->density + tu->density + tv->density;

    if (fabs(s) < 0.01) return;

    double d = 1.99 * (t->vel_x + t->vel_y - tu->vel_x - tv->vel_y) / s;

    t->vel_x -= d * t->density;
    t->vel_y -= d * t->density;
    tu->vel_x += d * tu->density;
    tv->vel_y += d * tv->density;
}

void gs_project_all(struct Tile * from, struct Tile * into)
{
    for (index_t x = 0; x < WIDTH; x++)
    for (index_t y = 0; y < HEIGHT; y++)
    {
        gs_project_tile(from, into, x, y);
    }
}

double density_between(struct Tile * board, index_t i, index_t j, index_t i_, index_t j_)
{
    index_t sign_i = sign(i_ - i);
    index_t sign_j = sign(j_ - j);

    double out = tile(board, i_, j_).density;

    for ( index_t ii = i; ii != i_; ii += sign_i ) out *= tile(board, ii, j).density;
    for ( index_t jj = j; jj < j_; jj += sign_j ) out *= tile(board, i_, jj).density;

    return out;

}

void advect(struct Tile * from, struct Tile * into, index_t i, index_t j, index_t i_, index_t j_, double frac)
{

    struct Tile * f = &(tile(from, i, j));
    struct Tile * x = &(tile(into, i, j));
    struct Tile * y = &(tile(into, i_, j_));

    double velx = frac * f->vel_x;
    double vely = frac * f->vel_y;
    double temp = frac * f->temp;

    double d = density_between(from, i, j, i_, j_);

    x->temp -= temp * d;
    x->vel_x -= velx * d;
    x->vel_y -= vely * d;

    y->vel_x += velx * d;
    y->vel_y += vely * d;
    y->temp += temp * d;
}

void advect_tile(struct Tile * from, struct Tile * into, index_t i, index_t j)
{
    double vel_x = delta_time * tile(from, i, j).vel_x;
    double vel_y = delta_time * tile(from, i, j).vel_y;
    
    index_t i_ = i + vel_x;
    index_t j_ = j + vel_y;

    double frac_i = (i + vel_x) - i_;
    double frac_j = (j + vel_y) - j_;

    double i0_part = (1 - frac_i) * vel_x;
    double i1_part = frac_i * vel_x;

    double j0_part = (1 - frac_j) * vel_y;
    double j1_part = frac_j * vel_y;

    advect(from, into, i, j, i_,     j_,     (1 - frac_i) * (1 - frac_j));
    advect(from, into, i, j, i_ + 1, j_,     frac_i       * (1 - frac_j));
    advect(from, into, i, j, i_,     j_ + 1, (1 - frac_i) * frac_j      );
    advect(from, into, i, j, i_ + 1, j_ + 1, frac_i       * frac_j      );
}

void advect_all(struct Tile * from, struct Tile * into)
{
    for (index_t x = 0; x < WIDTH; x++) 
    for (index_t y = 0; y < HEIGHT; y++)
    {
        advect_tile(from, into, x, y);
    }
}


void print_board(struct Tile * board)
{
    printf("\n\n FLUID SIM \n\n\n");

    for (index_t y = HEIGHT - 1; y >= 0 ; y--)
    {
        for (index_t x = 0; x < WIDTH; x++) 
        {
            struct Tile t = tile(board, x, y);

            if (t.density < 0.5) {printf("@"); continue;}
            printf( "%c", tileset(sqrt(t.temp + 0.001)) );
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

void external_consts(struct Tile * inplace) 
{
    for (int i = 0; i < WIDTH * HEIGHT; i++) 
    {
        inplace[i].vel_y += delta_time * inplace[i].temp;
        inplace[i].temp *= pow(0.99, delta_time);
        inplace[i].vel_y *= (inplace[i].vel_y > 0.5 * HEIGHT) ? 0 : 1;
        inplace[i].vel_x *= (inplace[i].vel_x > 0.5 * WIDTH) ? 0 : 1;
    }

    tile(inplace, WIDTH/2, HEIGHT/4).temp = 1;
}

int main() 
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
        tile(board, i, j).pressure = 0.0;
    }

    memcpy(board_, board, board_size);

    for (int i = 0; i < 100; i++)
    {
        print_board(board);

        for (int k = 0; k < 1 / delta_time; k++) 
        {
            external_consts(board);
            for (int j = 0; j < 20; j++) gs_project_all(board, board);
            memcpy(board_, board, board_size);
            advect_all(board, board_);
            memcpy(board, board_, board_size);
        }

    }

    return 0;
}
