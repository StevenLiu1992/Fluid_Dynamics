
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



#define SWAP(x0, x) {float4 *tmp = x0; x0 = x; x = tmp;}

__device__ float4 operator+(const float4 &a, const float4 &b);
__device__ float3 operator+(const float3 &a, const float3 &b);
__device__ float3 operator-(const float3 &a, const float3 &b);
__device__ float4 operator-(const float4 &a, const float4 &b);
__device__ float4 operator*(const float  &a, const float4 &b);
__device__ float3 operator*(const float  &a, const float3 &b);
__device__ float operator*(const float3 &a, const float3 &b);
__device__ float3 operator/(const float3 &b, const float  &a);
__device__ float3 normalize(const float3 &a);


__device__ void
boundary_density_condition_k(float *v, int ex, int ey, int ez, int scale, size_t pitch);

__device__ void
boundary_condition_k(float4 *v, int ex, int ey, int ez, int scale, size_t pitch);

/////////////////////////////////////////////////////////
__global__ void
advect_k(float4 *v);
__global__ void
jacobi_diffuse_k(float4 *v, float4 *temp, float4 *b, float *d, float alpha, float rBeta, size_t pitch);
__global__ void
jacobi_k(float4 *v, float4 *temp, float4 *b, float *d, float alpha, float rBeta, size_t pitch);
__global__ void
divergence_k(float4 *d, float4 *v, size_t pitch);

__global__ void
gradient_k(float4 *v, float4 *p, float *l, size_t pitch);

__global__ void
force_k(float4 *v, float *d, size_t pitch);

__global__ void
exterapolation_k(float4 *v, float4 *temp,float *l);
///////////////////////////////////////////////////////////
__global__ void
advectParticles_k(float3 *particle, float4 *v,
int dx, int dy, int dz, float dt, int lb, size_t pitch);

__global__ void
advectParticles_Runge_Kutta_k(float3 *particle, float4 *v, size_t pitch);

__global__ void
advect_density_k(float *d, size_t pitch);
__global__ void
bc_k(float4 *b, size_t pitch, float scale);
__global__ void
bc_density_k(float *b, size_t pitch, float scale);
__global__ void
advect_levelset_k(float *ls, int dx, int dy, int dz, float dt, size_t pitch);

__global__ void
correctLevelset_first_k(float3 *p, float2 *con);

__global__ void
correctLevelset_second_k(float *ls, float2 *con);

__global__ void
raycasting_k(int maxx, int maxy, float *ls, float3 camera);

__global__ void
reinit_Levelset_k(float *ls);

__global__ void
add_source_k(float4 *v, float *d, float *l, int x, int y, int z, int size);


void setupTexture(void);
void bindTexture(void);
void unbindTexture(void);
void update_vel_texture(float4 *data, int w, int h, size_t pitch);
void update_temp_texture(float4 *data, int dimx, int dimy, size_t pitch);
void update_1f_texture(cudaArray *array_1d, float *data, int w, int h, size_t pitch);
void deleteTexture(void);
void bindTexturetoCudaArray(void);

#endif