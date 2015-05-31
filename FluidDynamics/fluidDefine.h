
#pragma once

#ifndef FLUIDDATA_H
#define FLUIDDATA_H

#define NX 16
#define NY 16
#define NZ 16
#define DT 0.09f
#define DS (NX*NY*NZ)

#define IX ( x , y , z ,NX,NY) ( ( x ) + ( (NX) * ( ( y ) +( z ) *(NY) ) ) )


#define THREAD_X 8
#define THREAD_Y 8
#define THREAD_Z 8

#define VISC 0.000025f 

#endif