#include "fluidKernel.cuh"

texture<float4, 3> texref;
texture<float4, 3> texref1;
static cudaArray * array = NULL;
static cudaArray * array1 = NULL;
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
	checkCudaErrors(cudaMalloc3DArray(&array1, &ca_descriptor, volumeSize));
	getLastCudaError("cudaMalloc failed");
}

void bindTexture(void)
{
	cudaBindTextureToArray(texref, array);
	getLastCudaError("cudaBindTexture failed");

	cudaBindTextureToArray(texref1, array1);
	getLastCudaError("cudaBindTexture failed");
}

void unbindTexture(void)
{
	cudaUnbindTexture(texref);
	cudaUnbindTexture(texref1);
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

__device__ float4 operator+(const float4 &a, const float4 &b) {

	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,a.w+b.w);

}

__device__ float4 operator-(const float4 &a, const float4 &b) {

	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);

}

__device__ float4 operator*(const float &a, const float4 &b) {

	return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);

}

__global__ void
advect_k(float4 *v, int dx, int dy, int dz, float dt, int lb, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16
	//lb = 16
	
	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x*8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y*8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z*8;

	if (ex != 0 && ex != dx&&ey != 0 && ey != dy&&ez != 0 && ez != dz){

		float4 velocity;
		float3 ploc;

		
		//	float3 texcoord = { ex, ey, ez };
		velocity = tex3D(texref, (float)ex, (float)ey, (float)ez);
		ploc.x = (ex + 0.5f) - dt * velocity.x / dx;
		ploc.y = (ey + 0.5f) - dt * velocity.y / dy;
		ploc.z = (ez + 0.5f) - dt * velocity.z / dz;

		velocity = tex3D(texref, ploc.x, ploc.y, ploc.z);

	//	v[ez*NX*NY + ey*NX + ex] = velocity;
		
		/*float4 *f = (float4 *)((char *)v + ez * pitch) + ey * dy + ex;
		(*f).x = velocity.x;
		(*f).y = velocity.y;
		(*f).z = velocity.z;*/
		
	}
	
}

__global__ void
jacobi_k(float4 *v, float4 *temp, float4 *b, float alpha, float rBeta,
int dx, int dy, int dz, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16
	//lb = 16

	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;


	if (ex < 1){
		ex = 1;
	}
	if (ex > dx - 2){
		ex = dx - 2;
	}
	if (ey < 1){
		ey = 1;
	}
	if (ey > dy - 2){
		ey = dy - 2;
	}
	if (ez < 1){
		ez = 1;
	}
	if (ez > dz - 2){
		ez = dz - 2;
	}

//	if (ex != 0 && ex != dx&&ey != 0 && ey != dy&&ez != 0 && ez != dz){
		

				////value b
				//float3 *p0 = (float3 *)
				//	((char *)b + ez * pitch) + ey * dy + ex;
				////left x
				//float3 *p1 = (float3 *)
				//	((char *)temp + ez * pitch) + ey * dy + (ex - 1);
				////right x
				//float3 *p2 = (float3 *)
				//	((char *)temp + ez * pitch) + ey * dy + (ex + 1);
				////top x
				//float3 *p3 = (float3 *)
				//	((char *)temp + ez * pitch) + (ey - 1) * dy + ex;
				////bottom x
				//float3 *p4 = (float3 *)
				//	((char *)temp + ez * pitch) + (ey + 1) * dy + ex;
				////front x
				//float3 *p5 = (float3 *)
				//	((char *)temp + (ez - 1) * pitch) + ey * dy + ex;
				////behind x
				//float3 *p6 = (float3 *)
				//	((char *)temp + (ez + 1) * pitch) + ey * dy + ex;

				//float3 *New = (float3 *)
				//	((char *)v + ez * pitch) + ey * dy + ex;
		//		for (int count = 0; count < amount; count++){

					//value b
					float4 p0 = b[ez*NX*NY+ey*NX+ex];
	//	float4 p0 = tex3D(texref, (float)ex-1, (float)ey, (float)ez);
					/*float4 p1 = tex3D(texref, (float)ex - 1, (float)ey, (float)ez);
					float4 p2 = tex3D(texref, (float)ex + 1, (float)ey, (float)ez);
					float4 p3 = tex3D(texref, (float)ex, (float)ey - 1, (float)ez);
					float4 p4 = tex3D(texref, (float)ex, (float)ey + 1, (float)ez);
					float4 p5 = tex3D(texref, (float)ex, (float)ey, (float)ez - 1);
					float4 p6 = tex3D(texref, (float)ex, (float)ey, (float)ez + 1);*/
					//left x
					float4 p1 = temp[ez*NX*NY + ey*NX + ex - 1];
					//right x
					float4 p2 = temp[ez*NX*NY + ey*NX + ex + 1];
					//top x
					float4 p3 = temp[ez*NX*NY + (ey-1)*NX + ex];
					//bottom x
					float4 p4 = temp[ez*NX*NY + (ey + 1)*NX + ex];
					//front x
					float4 p5 = temp[(ez-1)*NX*NY + ey*NX + ex];
					//behind x
					float4 p6 = temp[(ez + 1)*NX*NY + ey*NX + ex];

					/*float3 *New = (float3 *)
						((char *)v + ez * pitch) + ey * dy + ex;*/
			v[ez*NX*NY + ey*NX + ex] = rBeta * (p1 + p2 + p3 + p4 + p5 + p6 + alpha * p0);
			//		*New = rBeta * ((*p1) + (*p2) + (*p3) + (*p4) + (*p5) + (*p6) + alpha * (*p0));

				/*	float4 *t = v;
					v = temp;
					temp = t;*/
			//	}

//	}
}

__global__ void
divergence_k(float4 *d, float4 *v,
int dx, int dy, int dz, int lb, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16
	//lb = 16

	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;
	if (ex < 1){
		ex = 1;
	}
	if (ex > dx - 2){
		ex = dx - 2;
	}
	if (ey < 1){
		ey = 1;
	}
	if (ey > dy - 2){
		ey = dy - 2;
	}
	if (ez < 1){
		ez = 1;
	}
	if (ez > dz - 2){
		ez = dz - 2;
	}
//	if (ex != 0 && ex != dx&&ey != 0 && ey != dy&&ez != 0 && ez != dz){

		////left x
		//float4 *p1 = (float4 *)
		//	((char *)v + ez * pitch) + ey * dy + (ex - 1);
		////right x
		//float4 *p2 = (float4 *)
		//	((char *)v + ez * pitch) + ey * dy + (ex + 1);
		////top x
		//float4 *p3 = (float4 *)
		//	((char *)v + ez * pitch) + (ey - 1) * dy + ex;
		////bottom x
		//float4 *p4 = (float4 *)
		//	((char *)v + ez * pitch) + (ey + 1) * dy + ex;
		////front x
		//float4 *p5 = (float4 *)
		//	((char *)v + (ez - 1) * pitch) + ey * dy + ex;
		////behind x
		//float4 *p6 = (float4 *)
		//	((char *)v + (ez + 1) * pitch) + ey * dy + ex;

		float4 p1 = v[ez*NX*NY + ey*NX + ex - 1];
		//right x
		float4 p2 = v[ez*NX*NY + ey*NX + ex + 1];
		//top x
		float4 p3 = v[ez*NX*NY + (ey - 1)*NX + ex];
		//bottom x
		float4 p4 = v[ez*NX*NY + (ey + 1)*NX + ex];
		//front x
		float4 p5 = v[(ez - 1)*NX*NY + ey*NX + ex];
		//behind x
		float4 p6 = v[(ez + 1)*NX*NY + ey*NX + ex];

//		float4 *New = (float4 *)((char *)d + ez * pitch) + ey * dy + ex;

		float div = 0.5*((p2.x - p1.x) + (p4.y - p3.y) + (p6.z - p5.z));
		d[ez*NX*NY + ey*NX + ex].x = div;
		d[ez*NX*NY + ey*NX + ex].y = div;
		d[ez*NX*NY + ey*NX + ex].z = div;
		d[ez*NX*NY + ey*NX + ex].w = div;
				
		/*(*New).x = 0.5 * (p1.x - p2.x + (p3.y - p4.y) + (p5.z - p6.z));
		(*New).y = 0.5 * (p1.x - p2.x + (p3.y - p4.y) + (p5.z - p6.z));
		(*New).z = 0.5 * (p1.x - p2.x + (p3.y - p4.y) + (p5.z - p6.z));*/
		//(*New).y = 0.5 * (((*p1).x - (*p2).x) + ((*p3).y - (*p4).y) + ((*p5).z - (*p6).z));
		//(*New).z = 0.5 * (((*p1).x - (*p2).x) + ((*p3).y - (*p4).y) + ((*p5).z - (*p6).z));
//	}
		
	
}



__global__ void
gradient_k(float4 *v, float4 *p,
int dx, int dy, int dz, int lb, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16
	//lb = 16


	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;


	if (ex < 1){
		ex=1;
	}
	if (ex > dx-2){
		ex = dx-2;
	}
	if (ey < 1){
		ey = 1;
	}
	if (ey > dy-2){
		ey = dy-2;
	}
	if (ez < 1){
		ez = 1;
	}
	if (ez > dz-2){
		ez = dz-2;
	}
//	if (ex != 0 && ex != dx&&ey != 0 && ey != dy&&ez != 0 && ez != dz){
		////left x
		//float4 *p1 = (float4 *)
		//	((char *)p + ez * pitch) + ey * dy + (ex - 1);
		////right x
		//float4 *p2 = (float4 *)
		//	((char *)p + ez * pitch) + ey * dy + (ex + 1);
		////top x
		//float4 *p3 = (float4 *)
		//	((char *)p + ez * pitch) + (ey - 1) * dy + ex;
		////bottom x
		//float4 *p4 = (float4 *)
		//	((char *)p + ez * pitch) + (ey + 1) * dy + ex;
		////front x
		//float4 *p5 = (float4 *)
		//	((char *)p + (ez - 1) * pitch) + ey * dy + ex;
		////behind x
		//float4 *p6 = (float4 *)
		//	((char *)p + (ez + 1) * pitch) + ey * dy + ex;

		float4 p1 = p[ez*NX*NY + ey*NX + ex - 1];
		//right x
		float4 p2 = p[ez*NX*NY + ey*NX + ex + 1];
		//top x
		float4 p3 = p[ez*NX*NY + (ey - 1)*NX + ex];
		//bottom x
		float4 p4 = p[ez*NX*NY + (ey + 1)*NX + ex];
		//front x
		float4 p5 = p[(ez - 1)*NX*NY + ey*NX + ex];
		//behind x
		float4 p6 = p[(ez + 1)*NX*NY + ey*NX + ex];


		//float4 *New = (float4 *)((char *)v + ez * pitch) + ey * dy + ex;
		float4 vel = v[ez*NX*NY + ey*NX + ex];
		float4 grad;
		grad.x = 0.5*(p2.x - p1.x);
		grad.y = 0.5*(p4.x - p3.x);
		grad.z = 0.5*(p6.x - p5.x);

		v[ez*NX*NY + ey*NX + ex] = vel - grad;
		/*v[ez*NX*NY + ey*NX + ex].x = vel.x - 0.5 * ((p2.x - p1.x) + (p4.y - p3.y) + (p6.z - p5.z));
		v[ez*NX*NY + ey*NX + ex].y = vel.y - 0.5 * ((p2.x - p1.x) + (p4.y - p3.y) + (p6.z - p5.z));
		v[ez*NX*NY + ey*NX + ex].z = vel.z - 0.5 * ((p2.x - p1.x) + (p4.y - p3.y) + (p6.z - p5.z));*/
	//	float4 t = (*New) - 0.5 * (((*p2) - (*p1)) + ((*p4) - (*p3)) + ((*p6) - (*p5)));
	//	*New = t;
			
//	}
	
	
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
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;

	if (ex != 0 && ex != dx&&ey != 0 && ey != dy&&ez != 0 && ez != dz){
		int index = ez*dx*dy + ey*dx + ex;


		float3 position = particle[index];

		/*float4 *vloc = (float4 *)
			((char *)v + ez * pitch) + ey * dy + ex;*/
		float3 newPosition;

		float4 vloc = tex3D(texref, position.x * dx, position.y* dy, position.z* dz);

		newPosition.x = position.x + dt * vloc.x;
		newPosition.y = position.y + dt * vloc.y;
		newPosition.z = position.z + dt * vloc.z;



		/*newPosition.x = (float)ex / dx;
		newPosition.y = (float)ey / dy;
		newPosition.z = (float)ez / dz;*/
		particle[index] = newPosition;
	}
	
}

extern "C"
void advect(float4 *v, float4 *temp, int dx, int dy, int dz, float dt)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);

	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);

	updateTexture(v, NX, NY, tPitch_v);
	advect_k<<<block_size, threads_size >>>(v, dx, dy, dz, dt, NY / THREAD_Y, tPitch_v);

	getLastCudaError("advectVelocity_k failed.");
	
}

extern "C"
void diffuse(float4 *v, float4 *temp, int dx, int dy, int dz, float dt)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);

	float rdx = 1.f;
	float alpha = rdx / VISC / dt;
	float rBeta = 1 / (6 + rdx / VISC / dt);
	cudaMemcpy(temp, v, sizeof(float4) * DS, cudaMemcpyDeviceToDevice);
	for(int i=0;i<20;i++){
	//	updateTexture(temp, NX, NY, tPitch_v);
		//xNew, x, b, alpha, rBeta, dx, dy, dz, pitch;
		jacobi_k << <block_size, threads_size >> >(v, temp, temp, alpha, rBeta, dx, dy, dz, tPitch_v);
		SWAP(v, temp);
	//	updateTexture(temp, NX, NY, tPitch_v);
	}
	

	getLastCudaError("diffuse_k failed.");
}

extern "C"
void projection(float4 *v, float4 *temp, float4 *pressure, float4* divergence, int dx, int dy, int dz, float dt)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);

	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	
	
	divergence_k<<<block_size, threads_size >>>(divergence, v, dx, dy, dz, NY / THREAD_Y, tPitch_v);

	cudaMemset(pressure, 0, sizeof(float4)*NX*NY*NZ);
	
	for(int i = 0; i < 40; i++){
		jacobi_k<<<block_size, threads_size >>>(temp, pressure, divergence, -1, 1.f / 6, dx, dy, dz, tPitch_v);
		SWAP(pressure, temp);
	}
	gradient_k<<<block_size, threads_size >>>(v, pressure, dx, dy, dz, NY / THREAD_Y, tPitch_v);

	getLastCudaError("diffuse_k failed.");
}


extern "C"
void advectParticles(GLuint vbo, float4 *v, int dx, int dy, int dz, float dt)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);

	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);

	float3 *p;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&p, &num_bytes, cuda_vbo_resource);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	updateTexture(v, NX, NY, tPitch_v);
	advectParticles_k<<<block_size, threads_size >>>(p, v, dx, dy, dz, dt, NY / THREAD_Y, tPitch_v);
	getLastCudaError("advectParticles_k failed.");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");
}