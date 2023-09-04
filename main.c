#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <errno.h>    

/* msleep(): Sleep for the requested number of milliseconds. */
int msleep(long msec)
{
    struct timespec ts;
    int res;

    if (msec < 0)
    {
        errno = EINVAL;
        return -1;
    }

    ts.tv_sec = msec / 1000;
    ts.tv_nsec = (msec % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);

    return res;
}


#define WIDTH 128
#define HEIGHT 32

#define BOUNDARY ((struct Tile *) ((struct Tile []) {(struct Tile) {0,0,0,0,0}}))
#define tile(board, x, y) (board)[ ((uint64_t) ((y + HEIGHT) % HEIGHT) * WIDTH) + ((x + WIDTH) % WIDTH) ]

#define index_t int
#define board_size (sizeof(struct Tile) * WIDTH * HEIGHT)

#define delta_time 0.1

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



void gs_project_tile(struct Tile * from, struct Tile * into, index_t i, index_t j)
{ 
    double in_x = tile(from, i, j).vel_x;
    double in_y = tile(from, i, j).vel_y;
    double out_x = tile(from, i + 1, j).vel_x;
    double out_y = tile(from, i, j + 1).vel_y;

    double d_in = tile(from, i, j).density;
    double d_out_x = tile(from, i + 1, j).density;
    double d_out_y = tile(from, i, j + 1).density;

    double s = d_in + d_in + d_out_x + d_out_y;

    if (fabs(s) < 0.1) return;

    double over_relax = 1.99; // only works with boundary conds.
    double d = over_relax * (in_x + in_y - out_x - out_y) / s;

    tile(into, i, j).pressure = d;

    tile(into, i, j).vel_x -= d * d_in;
    tile(into, i, j).vel_y -= d * d_in;
    tile(into, i + 1, j).vel_x += d * d_out_x;
    tile(into, i, j + 1).vel_y += d * d_out_y;
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
    index_t delta_i = fabs((int) i - i_);
    index_t delta_j = fabs((int) j - j_);

    double out = tile(board, max(i, i_), max(j, j_)).density;

    for ( int ii = min(i, i_); ii < max(i, i_); ii++ ) out *= tile(board, ii, j).density;
    for ( int jj = min(j, j_); jj < max(j, j_); jj++ ) out *= tile(board, i, jj).density;

    // if (delta_i > delta_j)
    // {
    //     double j_grad = ( (double) j_ - j ) / ( (double) i_ - i );
    //     for ( int ii = 0; ii <= max(i, i_) - min(i, i_); ii++ )
    //     {
    //         out *= tile( board, (int) (i + ii * sign(i_ - i)), (int) (j + ii * sign(i_ - i) * j_grad) ).density;
    //     }
    // }
    // else
    // {
    //     double i_grad = ( (double) i_ - i ) / ( (double) j_ - j );
    //     for ( int jj = 0; jj <= max(j, j_) - min(j, j_); jj++ )
    //     {
    //         out *= tile( board, (int) (i + jj * sign(j_ - j) * i_grad), (int) (j + jj * sign(j_ - j)) ).density;
    //     }
    // }

    return out;

}

void advect(struct Tile * from, struct Tile * into, index_t i, index_t j, index_t i_, index_t j_, double frac)
{

    double velx = frac * tile(from, i, j).vel_x;
    double vely = frac * tile(from, i, j).vel_y;
    double temp = frac * tile(from, i, j).temp;

    // if ((i_ < 0) || (i_ >= WIDTH) || (j_ < 0) || (j_ >= HEIGHT)) return;

    double d = density_between(from, i, j, i_, j_);

    tile(into, i, j).temp -= temp * d;
    tile(into, i, j).vel_x -= velx * d;
    tile(into, i, j).vel_y -= vely * d;

    tile(into, i_, j_).vel_x += velx * d;
    tile(into, i_, j_).vel_y += vely * d;
    tile(into, i_, j_).temp += temp * d;
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
            // printf( "%.2f\t", t.vel_x );

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

    // tile(inplace, 1, HEIGHT/4).vel_x = (rand() % 15) - 7;
    // tile(inplace, 1, HEIGHT/4).vel_y = 0;
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
        // tile(board, i, j).density *= ! ( i > WIDTH / 4 && j > HEIGHT / 4 && i < WIDTH / 2 && j < HEIGHT / 2 );
        tile(board, i, j).density *= ! ( pow(i - WIDTH/2, 2) + pow((j * 2) - 2 * 0.7 * HEIGHT, 2) < pow(7, 2) );
        
        tile(board, i, j).vel_x = 0.0;
        tile(board, i, j).vel_y = 0.0;
        tile(board, i, j).temp = 0.0;
        tile(board, i, j).pressure = 0.0;
    }

    memcpy(board_, board, board_size);

    for (int i = 0; i < 1000; i++)
    {
        print_board(board);
        // msleep(25);

        for (int k = 0; k < 1 / delta_time; k++) 
        {
            external_consts(board);

            for (int j = 0; j < 30; j++) gs_project_all(board, board);

            memcpy(board_, board, board_size);

            advect_all(board, board_);

            memcpy(board, board_, board_size);
        }

        
        
    }

    return 0;
}
