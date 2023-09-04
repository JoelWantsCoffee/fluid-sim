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

#define tile(board, x, y) (board)[ ((uint64_t) y * WIDTH) + x ]
#define index_t int
#define board_size (sizeof(struct Tile) * WIDTH * HEIGHT)

#define delta_time 0.05

#define tileset(i) (" .,:;ilw8WM")[(int) ((i < 0)?0:((i*10 < 10)?i*10:10))]

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
    double in_x = ( i >= 0 && j >= 0 ) ? tile(from, i, j).vel_x : 0.0;
    double in_y = ( i >= 0 && j >= 0 ) ? tile(from, i, j).vel_y : 0.0;
    double out_x = ( j >= 0 && i + 1 < WIDTH ) ? tile(from, i + 1, j).vel_x : 0.0;
    double out_y = ( i >= 0 && j + 1 < HEIGHT ) ? tile(from, i, j + 1).vel_y : 0.0;

    double d_in = ( i >= 0 && j >= 0 ) ? tile(from, i, j).density : 0.0;
    double d_out_x = ( j >= 0 && i + 1 < WIDTH ) ? tile(from, i + 1, j).density : 0.0;
    double d_out_y = ( i >= 0 && j + 1 < HEIGHT ) ? tile(from, i, j + 1).density : 0.0;

    double s = (i >= 0) * (1 - d_in) 
                + (j >= 0) * (1 - d_in)
                + (i + 1 < WIDTH) * (1 - d_out_x) 
                + (j + 1 < HEIGHT) * (1 - d_out_y);

    if (fabs(s) < 0.01) return;

    double over_relax = 1.9; // only works with boundary conds.
    double d = over_relax * (in_x + in_y - out_x - out_y) / s;

    if ( i >= 0 && j >= 0 ) tile(into, i, j).pressure = d;

    if ( i >= 0 && j >= 0 ) tile(into, i, j).vel_x -= d * (1 - d_in) * (1 - d_out_x);
    if ( i >= 0 && j >= 0 ) tile(into, i, j).vel_y -= d * (1 - d_in) * (1 - d_out_y);
    if ( j >= 0 && i + 1 <  WIDTH ) tile(into, i + 1, j).vel_x += d * (1 - d_in) * (1 - d_out_x);
    if ( i >= 0 && j + 1 < HEIGHT ) tile(into, i, j + 1).vel_y += d * (1 - d_in) * (1 - d_out_y);
}

void gs_project_all(struct Tile * from, struct Tile * into)
{
    for (index_t x = 0; x < WIDTH; x++)
    for (index_t y = 0; y < HEIGHT; y++)
    {
        gs_project_tile(from, into, x, y);
    }
}

void advect(struct Tile * from, struct Tile * into, index_t i, index_t j, index_t i_, index_t j_, double frac)
{

    double velx = frac * tile(from, i, j).vel_x;
    double vely = frac * tile(from, i, j).vel_y;
    double temp = frac * tile(from, i, j).temp;

    if ((i_ < 0) || (i_ >= WIDTH) || (j_ < 0) || (j_ >= HEIGHT)) return;

    double d = tile(into, i, j).density;
    double d_ = tile(into, i_, j_).density;

    tile(into, i, j).temp -= temp * (1 - d) * (1 - d_);
    tile(into, i, j).vel_x -= velx * (1 - d) * (1 - d_);
    tile(into, i, j).vel_y -= vely * (1 - d) * (1 - d_);

    tile(into, i_, j_).vel_x += velx * (1 - d) * (1 - d_);
    tile(into, i_, j_).vel_y += vely * (1 - d) * (1 - d_);
    tile(into, i_, j_).temp += temp * (1 - d) * (1 - d_);
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
    printf("\n FLUID SIM \n\n");

    for (index_t x = 0; x < WIDTH + 2; x++) printf("#"); printf("\n");

    for (index_t y = HEIGHT - 1; y >= 0 ; y--)
    {
        printf("#");
        for (index_t x = 0; x < WIDTH; x++) 
        {
            struct Tile t = tile(board, x, y);

            printf( "%c", tileset(sqrt(t.temp + 0.001)) );
            // printf( "%c", tileset(0.5 * (fabs(t.vel_x) + fabs(t.vel_y))) );
            // printf( "%.2f\t", t.vel_x );

            // if ((fabs(t.vel_y) < 0.05) && (fabs(t.vel_x) < 0.05)) printf(" ");
            // else if (fabs(t.vel_y) < 0.05) printf("-");
            // else if (fabs(t.vel_x) < 0.05) printf("|");
            // else if ((fabs(t.vel_y) > 1) && (fabs(t.vel_x) > 1)) printf("+");
            // else if (t.vel_x * t.vel_y < 0) printf("/");
            // else printf("\\");
        }
        printf("#\n");
    }

    for (index_t x = 0; x < WIDTH + 2; x++) printf("#"); printf("\n");
}

void external_consts(struct Tile * inplace) {
    for (int i = 0; i < WIDTH * HEIGHT; i++) 
    {
        inplace[i].vel_y += delta_time * inplace[i].temp;
        inplace[i].temp *= pow(0.95, delta_time);
    }
}

int main() 
{
    struct Tile * board = malloc(board_size);
    struct Tile * board_ = malloc(board_size);

    for (int i = 0; i < WIDTH; i++) 
    for (int j = 0; j < HEIGHT; j++) 
    {
        tile(board, i, j).density = 0;// ( i > WIDTH / 2 ) && (j < HEIGHT / 2 );
        tile(board, i, j).vel_x = 0.0;
        tile(board, i, j).vel_y = 0.0;
        tile(board, i, j).temp = 0.0;
        tile(board, i, j).pressure = 0.0;
    }

    memcpy(board_, board, board_size);

    for (int i = 0; i < 1000; i++)
    {
        print_board(board);
        // msleep(50);

        for (int k = 0; k < 1 / delta_time; k++) 
        {
            tile(board, 0, HEIGHT/4).vel_x = sqrt(rand() % (20 * 20));
            tile(board, 0, HEIGHT/4).temp = (rand() % 10) / 5.0;
            external_consts(board);

            for (int j = 0; j < 30; j++) {
                gs_project_all(board, board);
                // memcpy(board, board_, board_size);
            }

            memcpy(board_, board, board_size);
            advect_all(board, board_);
            memcpy(board, board_, board_size);
        }

        
        
    }

    return 0;
}
