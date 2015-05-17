#include "fluidKernel.cuh"

texture<float3, 3> texref;
static cudaArray *array = NULL;
cudaChannelFormatDesc ca_descriptor;
cudaExtent volumeSize;

// Texture pitch
extern size_t tPitch;

void setupTexture(int x, int y, int z)
{
	// Wrap mode appears to be the new default
	texref.filterMode = cudaFilterModeLinear;
//	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float3>();
	volumeSize = make_cudaExtent(NX, NY, NZ);
	ca_descriptor = cudaCreateChannelDesc<float3>();
	ca_descriptor.x = NX;
	ca_descriptor.y = NY;
	ca_descriptor.z = NZ;
	cudaMalloc3DArray(&array, &ca_descriptor, volumeSize);
	getLastCudaError("cudaMalloc failed");
}

void bindTexture(void)
{
	cudaBindTextureToArray(texref, array);
	getLastCudaError("cudaBindTexture failed");
}

void unbindTexture(void)
{
	cudaUnbindTexture(texref);
}

void updateTexture(float3 *data, size_t dimx, size_t dimy, size_t pitch)
{
	cudaMemcpy3DParms cpy_params = { 0 };
	cpy_params.extent = volumeSize;
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.dstArray = array;
	cpy_params.srcPtr = make_cudaPitchedPtr((float3*)data, pitch, dimx, dimy);
	cudaMemcpy3D(&cpy_params);
	getLastCudaError("cudaMemcpy failed");
}

__device__ float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}
__device__ float3 operator*(const float &a, const float3 &b) {

	return make_float3(a * b.x, a * b.y, a * b.z);

}

__global__ void
advect_k(float3 *v, float3 *temp,
	int dx, int dy, int dz, float dt, int lb, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16
	//lb = 16
	
	// ex is the domain location in x for this thread
	int ex = threadIdx.x;
	
	// ez is the domain location in z for this thread
	int ez = blockIdx.y * 4 + blockIdx.x;

	float3 velocity, ploc;
	

	
	for (int i = 0; i < lb; i++)
	{
		// ey is the domain location in y for this thread
		int ey = threadIdx.y * lb + i;

		if (ey < dy)
		{
			
			float3 texcoord = { ex, ey, ez };
			velocity = tex3D(texref, texcoord);
			ploc.x = (ex + 0.5f) - (dt * velocity.x * dx);
			ploc.y = (ey + 0.5f) - (dt * velocity.y * dy);
			ploc.z = (ez + 0.5f) - (dt * velocity.z * dz);
		
			velocity = tex3D(texref, ploc);
			

			float3 *f = (float3 *)((char *)temp + ez * pitch) + ey * dy + ex;
			*f = velocity;
		}
	}
	
}

__global__ void
jacobi_k(float3 *v, float3 *temp, float alpha, float rBeta,
	int dx, int dy, int dz, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16
	//lb = 16

	// ex is the domain location in x for this thread
	int ex = threadIdx.x;

	// ez is the domain location in z for this thread
	int ez = blockIdx.y * 4 + blockIdx.x;


	for (int i = 0; i < lb; i++)
	{
		// ey is the domain location in y for this thread
		int ey = threadIdx.y * lb + i;

		if (ey < dy)
		{

			
			//basic
			float3 *p0 = (float3 *)
				((char *)temp + ez * pitch) + ey * dy + ex;
			//left
			float3 *p1 = (float3 *)
				((char *)temp + ez * pitch) + ey * dy + (ex-1);
			//right
			float3 *p2 = (float3 *)
				((char *)temp + ez * pitch) + ey * dy + (ex+1);
			//top
			float3 *p3 = (float3 *)
				((char *)temp + ez * pitch) + (ey-1) * dy + ex;
			//bottom
			float3 *p4 = (float3 *)
				((char *)temp + ez * pitch) + (ey+1) * dy + ex;
			//front
			float3 *p5 = (float3 *)
				((char *)temp + (ez-1) * pitch) + ey * dy + ex;
			//behind
			float3 *p6 = (float3 *)
				((char *)temp + (ez+1) * pitch) + ey * dy + ex;
			
			float3 *New = (float3 *)
				((char *)v + ez * pitch) + ey * dy + ex;
			*New = rBeta * ((*p1) + (*p2) + (*p3) + (*p4) + (*p5) + (*p6) + alpha * (*p0));
		}
	}
}


extern "C"
void advect(float2 *v, float *temp, int dx, int dy, int dz, float dt)
{
	dim3 block_size((dx / BLOCK_X) + (!(dx%BLOCK_X) ? 0 : 1), (dy / BLOCK_Y) + (!(dy%BLOCK_Y) ? 0 : 1));

	dim3 threads_size(THREAD_X, THREAD_Y);

	updateTexture(v, NX, NY, tPitch);
	advect_k<<<block_size, threads_size>>>(v, temp, dx, dy, dz, dt, NY / THREAD_Y, tPitch);

	getLastCudaError("advectVelocity_k failed.");
}