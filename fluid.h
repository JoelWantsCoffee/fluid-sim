// constants
#define WIDTH 256
#define HEIGHT 96
#define SOLVER_ITERATIONS 100
#define DELTA_TIME 0.05
#define TIME_PER_FRAME 0.25
#define FRAMES 1000
#define index_t int32_t

// useful functions
#define tile(board, x, y) (board)[ ((int) ((y + HEIGHT) % HEIGHT) * WIDTH) + ((x + WIDTH) % WIDTH) ]
#define tile_unsafe(board, x, y) (board)[ ((int) (y % HEIGHT) * WIDTH) + (x % WIDTH) ]
#define board_size (sizeof(struct Tile) * WIDTH * HEIGHT)
#define simd_board_size (sizeof(__m256) * (WIDTH/8 + 1) * (HEIGHT + 1))

// data for each grid position
struct Tile 
{
    float temp;
    float vel_x;
    float vel_y;
    float density;
    int flags;
    #define ADVECTED 1
};

void populate_simd(struct Tile * from, struct Tile * to, __m256 * precomp_s, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely, __m256 * temp);
void unpopulate_simd(struct Tile * from, struct Tile * to, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely, __m256 * temp);

#define NOTES "gpu final"
