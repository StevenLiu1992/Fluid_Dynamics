
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

void setupTexture(void);
void bindTexture(void);
void unbindTexture(void);
void update_vel_texture(float4 *data, int w, int h, size_t pitch);
void update_1f_texture(cudaArray *array_1d, float *data, int w, int h, size_t pitch);
void update_temp_texture(float4 *data, int dimx, int dimy, size_t pitch);
void deleteTexture(void);
void bindTexturetoCudaArray(void);

#define SWAP(x0, x) {float4 *tmp = x0; x0 = x; x = tmp;}


__global__ void
forces_k(float4 *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch);

__global__ void
advect_k(float4 *v);

__global__ void
advect_density_k(float *d, int dx, int dy, int dz, float dt, size_t pitch);

__global__ void
jacobi_k(float4 *v, float4 *temp, float4 *b, float alpha, float rBeta, size_t pitch);

__global__ void
divergence_k(float4 *d, float4 *v, size_t pitch);

__global__ void
gradient_k(float4 *v, float4 *p,
int dx, int dy, int dz, int lb, size_t pitch);

__global__ void
advectParticles_k(float3 *particle, float4 *v, float* d,
int dx, int dy, int dz, int lb, size_t pitch);


__global__ void
bc_k(float4 *b, size_t pitch, float scale);

__global__ void
bc_density_k(float *b, size_t pitch, float scale);

__global__ void
force_k(float4 *v, int dx, int dy, int dz, float dt, size_t pitch);

__global__ void
advect_levelset_k(float *ls, int dx, int dy, int dz, float dt, size_t pitch);

__global__ void
correctLevelset_first_k(float3 *p, float2 *con);

__global__ void
correctLevelset_second_k(float *ls, float2 *con);

__global__ void
raycasting_k(int maxx, int maxy, float *ls, float4 *intersection, float3 camera);
#endif