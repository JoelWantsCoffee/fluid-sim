// constants
#define WIDTH 224
#define HEIGHT 64
#define SOLVER_ITERATIONS 100
#define DELTA_TIME 0.05
#define FRAMES 300
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
};

void populate_simd(struct Tile * from, struct Tile * to, __m256 * precomp_s, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely);
void unpopulate_simd(struct Tile * from, struct Tile * to, __m256 * density, __m256 * from_velx, __m256 * from_vely, __m256 * to_velx, __m256 * to_vely);

#define NOTES "the details..."