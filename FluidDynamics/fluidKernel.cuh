
#ifndef __FLUIDS_KERNELS_CUH_
#define __FLUIDS_KERNELS_CUH_


#include "fluidDefine.h"
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>    // Helper functions for CUDA Error handling


// OpenGL Graphics includes
#include "Dependencies/glew/glew.h"
#include "Dependencies/freeglut/freeglut.h"

void setupTexture(int x, int y, int z);
void bindTexture(void);
void unbindTexture(void);
void updateTexture(float4 *data, int w, int h, size_t pitch);
void deleteTexture(void);

#define SWAP(x0, x) {float4 *tmp = x0; x0 = x; x = tmp;}


__global__ void
forces_k(float4 *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch);

__global__ void
advect_k(float4 *v, int dx, int dy, int dz, float dt, int lb, size_t pitch);

__global__ void
jacobi_k(float4 *v, float4 *temp, float4 *b, float alpha, float rBeta,
int dx, int dy, int dz, size_t pitch);

__global__ void
divergence_k(float4 *d, float4 *v,
int dx, int dy, int dz, int lb, size_t pitch);

__global__ void
gradient_k(float4 *v, float4 *p,
int dx, int dy, int dz, int lb, size_t pitch);

__global__ void
advectParticles_k(float3 *particle, float4 *v,
int dx, int dy, int dz, int lb, size_t pitch);

#endif