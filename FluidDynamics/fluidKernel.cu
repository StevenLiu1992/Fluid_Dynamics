#include "fluidKernel.cuh"

texture<float4, 3> texref;
static cudaArray * array = NULL;
cudaChannelFormatDesc ca_descriptor;
cudaExtent volumeSize;

// Texture pitch
extern size_t tPitch_v;
extern size_t tPitch_t;
extern size_t tPitch_p;
extern size_t tPitch_d;
// Particle data
extern GLuint vbo;                 // OpenGL vertex buffer object
extern struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange

void setupTexture(int x, int y, int z)
{
	// Wrap mode appears to be the new default
	texref.filterMode = cudaFilterModeLinear;
//	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float3>();
	volumeSize = make_cudaExtent(NX, NY, NZ);
	ca_descriptor = cudaCreateChannelDesc<float4>();
	/*ca_descriptor.x = NX;
	ca_descriptor.y = NY;
	ca_descriptor.z = NZ;*/
	checkCudaErrors(cudaMalloc3DArray(&array, &ca_descriptor, volumeSize));
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

void updateTexture(float4 *data, int dimx, int dimy, size_t pitch)
{
	cudaMemcpy3DParms cpy_params = { 0 };
	cpy_params.extent = volumeSize;
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.dstArray = array;
	cpy_params.srcPtr = make_cudaPitchedPtr((void*)data, dimx*sizeof(float4), dimx, dimy);
	checkCudaErrors(cudaMemcpy3D(&cpy_params));
	getLastCudaError("cudaMemcpy failed");
}

__device__ float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

__device__ float3 operator-(const float3 &a, const float3 &b) {

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

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

	if (ex != 0 && ex != dx&&ez != 0 && ez != dz){

		float4 velocity;
		float3 ploc;

		for (int i = 0; i < lb; i++)
		{
			// ey is the domain location in y for this thread
			int ey = threadIdx.y * lb + i;

			if (ey!=0 && ey != dy)
			{

				//	float3 texcoord = { ex, ey, ez };
				velocity = tex3D(texref, (float)ex, (float)ey, (float)ez);
				ploc.x = (ex + 0.5f) - (dt * velocity.x);
				ploc.y = (ey + 0.5f) - (dt * velocity.y);
				ploc.z = (ez + 0.5f) - (dt * velocity.z);

				velocity = tex3D(texref, ploc.x, ploc.y, ploc.z);


				float3 *f = (float3 *)((char *)temp + ez * pitch) + ey * dy + ex;
				(*f).x = velocity.x;
				(*f).y = velocity.y;
				(*f).z = velocity.z;
			}
		}
	}
	
}

__global__ void
jacobi_k(float3 *v, float3 *temp, float3 *b, float alpha, float rBeta,
int dx, int dy, int dz, int lb, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16
	//lb = 16

	// ex is the domain location in x for this thread
	int ex = threadIdx.x;

	// ez is the domain location in z for this thread
	int ez = blockIdx.y * 4 + blockIdx.x;

	if (ex != 0 && ex != dx&&ez != 0 && ez != dz){
		for (int i = 0; i < lb; i++){

			// ey is the domain location in y for this thread
			int ey = threadIdx.y * lb + i;

			if (ey!=0 && ey != dy){

				//value b
				float3 *p0 = (float3 *)
					((char *)b + ez * pitch) + ey * dy + ex;
				//left x
				float3 *p1 = (float3 *)
					((char *)temp + ez * pitch) + ey * dy + (ex - 1);
				//right x
				float3 *p2 = (float3 *)
					((char *)temp + ez * pitch) + ey * dy + (ex + 1);
				//top x
				float3 *p3 = (float3 *)
					((char *)temp + ez * pitch) + (ey - 1) * dy + ex;
				//bottom x
				float3 *p4 = (float3 *)
					((char *)temp + ez * pitch) + (ey + 1) * dy + ex;
				//front x
				float3 *p5 = (float3 *)
					((char *)temp + (ez - 1) * pitch) + ey * dy + ex;
				//behind x
				float3 *p6 = (float3 *)
					((char *)temp + (ez + 1) * pitch) + ey * dy + ex;

				float3 *New = (float3 *)
					((char *)v + ez * pitch) + ey * dy + ex;
				*New = rBeta * ((*p1) + (*p2) + (*p3) + (*p4) + (*p5) + (*p6) + alpha * (*p0));
			}
		}
	}
}

__global__ void
divergence_k(float3 *d, float3 *v,
int dx, int dy, int dz, int lb, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16
	//lb = 16

	// ex is the domain location in x for this thread
	int ex = threadIdx.x;

	// ez is the domain location in z for this thread
	int ez = blockIdx.y * 4 + blockIdx.x;

	if (ex != 0 && ex != dx&&ez != 0 && ez != dz){
		for (int i = 0; i < lb; i++)
		{
			// ey is the domain location in y for this thread
			int ey = threadIdx.y * lb + i;

			if (ey != 0 && ey != dy){

				//left x
				float3 *p1 = (float3 *)
					((char *)v + ez * pitch) + ey * dy + (ex - 1);
				//right x
				float3 *p2 = (float3 *)
					((char *)v + ez * pitch) + ey * dy + (ex + 1);
				//top x
				float3 *p3 = (float3 *)
					((char *)v + ez * pitch) + (ey - 1) * dy + ex;
				//bottom x
				float3 *p4 = (float3 *)
					((char *)v + ez * pitch) + (ey + 1) * dy + ex;
				//front x
				float3 *p5 = (float3 *)
					((char *)v + (ez - 1) * pitch) + ey * dy + ex;
				//behind x
				float3 *p6 = (float3 *)
					((char *)v + (ez + 1) * pitch) + ey * dy + ex;

				float3 *New = (float3 *)
					((char *)d + ez * pitch) + ey * dy + ex;
				(*New).x = 0.5 * (((*p1).x - (*p2).x) + ((*p3).y - (*p4).y) + ((*p5).z - (*p6).z));
				(*New).y = 0.5 * (((*p1).x - (*p2).x) + ((*p3).y - (*p4).y) + ((*p5).z - (*p6).z));
				(*New).z = 0.5 * (((*p1).x - (*p2).x) + ((*p3).y - (*p4).y) + ((*p5).z - (*p6).z));
			}
		}
	}
}



__global__ void
gradient_k(float3 *v, float3 *p,
int dx, int dy, int dz, int lb, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16
	//lb = 16

	// ex is the domain location in x for this thread
	int ex = threadIdx.x;

	// ez is the domain location in z for this thread
	int ez = blockIdx.y * 4 + blockIdx.x;

	if (ex != 0 && ex != dx&&ez != 0 && ez != dz){
		for (int i = 0; i < lb; i++)
		{
			// ey is the domain location in y for this thread
			int ey = threadIdx.y * lb + i;

			if (ey != 0 && ey != dy){
				//left x
				float3 *p1 = (float3 *)
					((char *)p + ez * pitch) + ey * dy + (ex - 1);
				//right x
				float3 *p2 = (float3 *)
					((char *)p + ez * pitch) + ey * dy + (ex + 1);
				//top x
				float3 *p3 = (float3 *)
					((char *)p + ez * pitch) + (ey - 1) * dy + ex;
				//bottom x
				float3 *p4 = (float3 *)
					((char *)p + ez * pitch) + (ey + 1) * dy + ex;
				//front x
				float3 *p5 = (float3 *)
					((char *)p + (ez - 1) * pitch) + ey * dy + ex;
				//behind x
				float3 *p6 = (float3 *)
					((char *)p + (ez + 1) * pitch) + ey * dy + ex;

				float3 *New = (float3 *)
					((char *)v + ez * pitch) + ey * dy + ex;
				float3 t = (*New) - 0.5 * (((*p2) - (*p1)) + ((*p4) - (*p3)) + ((*p6) - (*p5)));
				*New = t;
			}
		}
	}
	
}

__global__ void
advectParticles_k(float3 *particle, float4 *v,
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


	for (int i = 0; i < lb; i++)
	{
		// ey is the domain location in y for this thread
		int ey = threadIdx.y * lb + i;

		if (ey < dy)
		{
			int index = ez*dx*dy + ey*dx + ex;
			
	
			float3 position = particle[index];

			float4 *v = (float4 *)
				((char *)v + ez * pitch) + ey * dy + ex;
			float3 newPosition;
			/*
			newPosition.x = position.x + dt * (*v).x;
			newPosition.y = position.y + dt * (*v).y;
			newPosition.z = position.z + dt * (*v).z;
			*/
			newPosition.x = (float)ex/dx;
			newPosition.y = (float)ey / dy;
			newPosition.z = (float)ez / dz+0.05;
			particle[index] = newPosition;
		}
	}
}

extern "C"
void advect(float4 *v, float4 *temp, int dx, int dy, int dz, float dt)
{
	//(dx / BLOCK_X) + (!(dx%BLOCK_X) ? 0 : 1), (dy / BLOCK_Y) + (!(dy%BLOCK_Y) ? 0 : 1)
	dim3 block_size(4,4);

	dim3 threads_size(THREAD_X, THREAD_Y);

	updateTexture(v, NX, NY, tPitch_v);
	advect_k<<<block_size, threads_size >>>((float3*)v, (float3*)temp, dx, dy, dz, dt, NY / THREAD_Y, tPitch_v);

	getLastCudaError("advectVelocity_k failed.");
	
}

extern "C"
void diffuse(float3 *v, float3 *temp, int dx, int dy, int dz, float dt)
{
	dim3 block_size(4, 4);

	dim3 threads_size(THREAD_X, THREAD_Y);
	for(int i=0;i<20;i++){
		//xNew, x, b, alpha, rBeta, dx, dy, dz, pitch;
		jacobi_k <<<block_size, threads_size >>>(v, temp, temp, 1 / VISC / dt, 1 / (6 + 1 / VISC / dt), dx, dy, dz, NY / THREAD_Y, tPitch_v);
		SWAP(v,temp);
	}
	

	getLastCudaError("diffuse_k failed.");
}

extern "C"
void projection(float3 *v, float3 *temp, float3 *pressure, float3* divergence, int dx, int dy, int dz, float dt)
{
	dim3 block_size(4, 4);

	dim3 threads_size(THREAD_X, THREAD_Y);
	
	
	divergence_k<<<block_size, threads_size >>>(divergence, v, dx, dy, dz, NY / THREAD_Y, tPitch_v);
	memset(pressure, 0, sizeof(float3) * dx*dy*dz);
	for(int i = 0; i < 40; i++){
		jacobi_k<<<block_size, threads_size >>>(temp, pressure, divergence, -1, 1 / 6, dx, dy, dz, NY / THREAD_Y, tPitch_v);
		SWAP(pressure, temp);
	}
	gradient_k<<<block_size, threads_size >>>(v, pressure, dx, dy, dz, NY / THREAD_Y, tPitch_v);

	getLastCudaError("diffuse_k failed.");
}


extern "C"
void advectParticles(GLuint vbo, float4 *v, int dx, int dy, int dz, float dt)
{
	dim3 block_size(4,4);

	dim3 threads_size(THREAD_X, THREAD_Y);

	float3 *p;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&p, &num_bytes,
		cuda_vbo_resource);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	advectParticles_k<<<block_size, threads_size >>>(p, v, dx, dy, dz, dt, NY / THREAD_Y, tPitch_v);
	getLastCudaError("advectParticles_k failed.");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");
}