
#pragma once

#ifndef FLUIDDATA_H
#define FLUIDDATA_H

#define NX 64
#define NY 64
#define NZ 8
#define DT 0.09f
#define DS (NX*NY*NZ)

#define IX ( x , y , z ,NX,NY) ( ( x ) + ( (NX) * ( ( y ) +( z ) *(NY) ) ) )
#define BLOCK_X 64
#define BLOCK_Y 64

#define THREAD_X 8
#define THREAD_Y 8
#define THREAD_Z 8

#define VISC 0.0025f 

#endif