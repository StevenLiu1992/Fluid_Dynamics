
#pragma once

#ifndef FLUIDDATA_H
#define FLUIDDATA_H

#define NX 32
#define NY 32
#define NZ 32

#define LNX 64
#define LNY 64
#define LNZ 64

#define PAMOUNT 64*64*64
#define TI (LNX/NX)

#define LDS LNX*LNY*LNZ

#define DT 0.1f
#define DS (NX*NY*NZ)

#define IX ( x , y , z ,NX,NY) ( ( x ) + ( (NX) * ( ( y ) +( z ) *(NY) ) ) )


#define THREAD_X 8
#define THREAD_Y 8
#define THREAD_Z 8

#define VISC 0.00025f 

#endif