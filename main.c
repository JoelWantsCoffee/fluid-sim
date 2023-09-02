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

#define tile(board, x, y) (board)[ (((uint64_t) y) * WIDTH) + x ]
#define index_t int16_t
#define board_size (sizeof(struct Tile) * WIDTH * HEIGHT)

#define delta_time 0.1

#define tileset(i) ("##`.,:ilwW#####" + 2)[(int) ((i < 0)?0:((i*7 < 7)?i*7:7))]

struct Tile 
{
    float vel_x;
    float vel_y;
    float density;
    float temp;
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
    float in_x = (i >= 0) ? tile(from, i, j).vel_x : 0.0;
    float in_y = (j >= 0) ? tile(from, i, j).vel_y : 0.0;
    float out_x = (i + 1 < WIDTH ) ? tile(from, i + 1, j).vel_x : 0.0;
    float out_y = (j + 1 < HEIGHT) ? tile(from, i, j + 1).vel_y : 0.0;

    float over_relax = 1.4; // only works with boundary conds.
    float d = over_relax * (in_x + in_y - out_x - out_y) / (0.0 + (i >= 0) + (j >= 0) + (i + 1 < WIDTH) + (j + 1 < HEIGHT));

    if ( i >= 0 ) tile(into, i, j).vel_x -= d;
    if ( j >= 0 ) tile(into, i, j).vel_y -= d;
    if ( i + 1 <  WIDTH ) tile(into, i + 1, j).vel_x += d;
    if ( j + 1 < HEIGHT ) tile(into, i, j + 1).vel_y += d;
}

void gs_project_all(struct Tile * from, struct Tile * into)
{
    for (index_t x = 0; x < WIDTH; x++)
    for (index_t y = 0; y < HEIGHT; y++)
    {
        gs_project_tile(from, into, x, y);
    }
}

void advect(struct Tile * from, struct Tile * into, index_t i, index_t j, index_t i_, index_t j_, float frac)
{
    if ((i_ < 0) || (i_ >= WIDTH) || (j_ < 0) || (j_ >= HEIGHT)) return;

    float velx = frac * tile(from, i, j).vel_x;
    float vely = frac * tile(from, i, j).vel_y;
    float temp = frac * tile(from, i, j).temp;

    tile(into, i_, j_).vel_x += velx;
    tile(into, i_, j_).vel_y += vely;
    tile(into, i_, j_).temp += temp;

    tile(into, i, j).vel_x -= velx;
    tile(into, i, j).vel_y -= vely;
    tile(into, i, j).temp -= temp;
}

void advect_tile(struct Tile * from, struct Tile * into, index_t i, index_t j)
{
    float vel_x = delta_time * tile(from, i, j).vel_x;
    float vel_y = delta_time * tile(from, i, j).vel_y;
    
    index_t i_ = i + vel_x;
    index_t j_ = j + vel_y;

    float frac_i = (i + vel_x) - i_;
    float frac_j = (j + vel_y) - j_;

    float i0_part = (1 - frac_i) * vel_x;
    float i1_part = frac_i * vel_x;

    float j0_part = (1 - frac_j) * vel_y;
    float j1_part = frac_j * vel_y;

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
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        inplace[i].vel_y += delta_time * inplace[i].temp;
        inplace[i].temp *= pow(0.99, delta_time);
    }
}

int main() 
{
    struct Tile * board = malloc(board_size);
    struct Tile * board_ = malloc(board_size);

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        board[i].density = 0.0;
        board[i].vel_x = 0.0;
        board[i].vel_y = 0.0;
        board[i].temp = 0.0;
    }

    memcpy(board_, board, board_size);

    for (int i = 0; i < 1000; i++)
    {
        print_board(board);
        msleep(50);

        for (int k = 0; k < 1 / delta_time; k++) 
        {
            tile(board, 0, HEIGHT/4).vel_x += 10;
            tile(board, 0, HEIGHT/4).temp = 1;
            external_consts(board);

            for (int j = 0; j < 50; j++) {
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
