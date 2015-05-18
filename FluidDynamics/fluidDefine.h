#define NX 64
#define NY 64
#define NZ 16
#define DS (NX*NY*NZ)

#define IX ( x , y , z ,NX,NY) ( ( x ) + ( (NX) * ( ( y ) +( z ) *(NY) ) ) )
#define BLOCK_X 64
#define BLOCK_Y 64

#define THREAD_X 64
#define THREAD_Y 4
#define VISC 0.0025f 