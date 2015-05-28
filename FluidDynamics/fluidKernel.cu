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
extern GLuint vbo2;                 // OpenGL vertex buffer object
extern struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
extern struct cudaGraphicsResource *cuda_vbo_resource1; // handles OpenGL-CUDA exchange

void setupTexture(int x, int y, int z)
{
	// Wrap mode appears to be the new default
	texref.filterMode = cudaFilterModeLinear;
//	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float3>();
	volumeSize = make_cudaExtent(NX, NY, NZ);
	ca_descriptor = cudaCreateChannelDesc<float4>();
	ca_descriptor.x = NX;
	ca_descriptor.y = NY;
	ca_descriptor.z = NZ;
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

__device__ float4 operator+(const float4 &a, const float4 &b) {

	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,a.w+b.w);

}

__device__ float4 operator-(const float4 &a, const float4 &b) {

	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);

}

__device__ float4 operator*(const float &a, const float4 &b) {

	return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);

}

//__device__ void
//boundary_k(float4 *v, int ex,int ey, int ez, float3 offset, float scale, size_t pitch, int type){
//	int pitch0 = pitch / sizeof(float4);
//	int ex0 = ex + offset.x;
//	int ey0 = ey + offset.y;
//	int ez0 = ez + offset.z;
//	//-1 * v[ez0*pitch0 + ey0*dx + ex0]
//	if (scale==-1&&ex==(NX-1)&&ey!=(dy-1))
//		v[ez0*pitch0 + ey0*dx + ex0] = make_float4(0, 0.01, 0, 0);
//
//	/*float4 *Velocity = (float4 *)((char *)v + ez * pitch) + ey * dy + ex;
//	float4 *Velocity0 = (float4 *)((char *)v + ez0 * pitch) + ey0 * dy + ex0;
//	*Velocity = scale * (*Velocity0);*/
//}

__device__ void
boundary_condition_k(float4 *v, int ex, int ey, int ez, int scale, size_t pitch){
	int pitch0 = pitch / sizeof(float4);
	if (ex == 0){
	//	float3 offset = make_float3(1, 0, 0);
		v[ez*pitch0 + ey*NX + ex].x = scale * v[ez*pitch0 + ey*NX + ex + 1].x;
	}
	if (ex == (NX - 1)){
		v[ez*pitch0 + ey*NX + ex].x = -1 * v[ez*pitch0 + ey*NX + ex - 1].x;
	}
	if (ey == 0){
		v[ez*pitch0 + ey*NX + ex].y = scale * v[ez*pitch0 + (ey + 1)*NX + ex].y;
	}
	if (ey == (NY - 1)){
		v[ez*pitch0 + ey*NX + ex].y = scale * v[ez*pitch0 + (ey - 1)*NX + ex].y;
	}
	if (ez == 0){
		v[ez*pitch0 + ey*NX + ex].z = scale * v[(ez + 1)*pitch0 + ey*NX + ex].z;
	}
	if (ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex].z = scale * v[(ez - 1)*pitch0 + ey*NX + ex].z;
	}
	/*if (ex == 0 && ey == 0 && ez == 0){
		float3 offset = { 1, 1, 1 };
		boundary_k(v, dx, dy, dz, ex, ey, ez, offset, scale, pitch);
	}
	if (ex == (dx - 1) && ey == 0 && ez == 0){
		float3 offset = { -1, 1, 1 };
		boundary_k(v, dx, dy, dz, ex, ey, ez, offset, scale, pitch);
	}
	if (ex == 0 && ey == (dy - 1) && ez == 0){
		float3 offset = { 1, -1, 1 };
		boundary_k(v, dx, dy, dz, ex, ey, ez, offset, scale, pitch);
	}
	if (ex == 0 && ey == 0 && ez == (dz - 1)){
		float3 offset = { 1, 1, -1 };
		boundary_k(v, dx, dy, dz, ex, ey, ez, offset, scale, pitch);
	}

	if (ex == (dx - 1) && ey == (dy - 1) && ez == 0){
		float3 offset = { -1, -1, 1 };
		boundary_k(v, dx, dy, dz, ex, ey, ez, offset, scale, pitch);
	}
	if (ex == (dx - 1) && ey == 0 && ez == (dz - 1)){
		float3 offset = { -1, 1, -1 };
		boundary_k(v, dx, dy, dz, ex, ey, ez, offset, scale, pitch);
	}
	if (ex == 0 && ey == (dy - 1) && ez == (dz - 1)){
		float3 offset = { 1, -1, -1 };
		boundary_k(v, dx, dy, dz, ex, ey, ez, offset, scale, pitch);
	}
	if (ex == (dx - 1) && ey == (dy - 1) && ez == (dz - 1)){
		float3 offset = { -1, -1, -1 };
		boundary_k(v, dx, dy, dz, ex, ey, ez, offset, scale, pitch);
	}*/

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

	if ((ex != 0) && (ex != (dx-1)) && (ey != 0) && (ey != (dy-1)) && (ez != 0) && (ez != (dz-1))){

		float4 velocity;
		float3 ploc;

		
		//	float3 texcoord = { ex, ey, ez };
		velocity = tex3D(texref, (float)ex, (float)ey, (float)ez);
		ploc.x = (ex + 0.5f) - dt * velocity.x / dx;
		ploc.y = (ey + 0.5f) - dt * velocity.y / dy;
		ploc.z = (ez + 0.5f) - dt * velocity.z / dz;





		velocity = tex3D(texref, ploc.x, ploc.y, ploc.z);

	//	v[ez*pitch + ey*NX + ex] = velocity;

		float4 *Velocity_field = (float4 *)((char *)v + ez * pitch) + ey * dy + ex;
		(*Velocity_field) = velocity;	
	}

	else{
	//	boundary_condition_k(v, dx, dy, dz, ex, ey, ez, -1, pitch);
	}
	__syncthreads();
}

__global__ void
jacobi_k(float4 *v, float4 *temp, float4 *b, float alpha, float rBeta,
int dx, int dy, int dz, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16

	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;


	//if (ex < 1){
	//	ex = 1;
	//}
	//if (ex > dx - 2){
	//	ex = dx - 2;
	//}
	//if (ey < 1){
	//	ey = 1;
	//}
	//if (ey > dy - 2){
	//	ey = dy - 2;
	//}
	//if (ez < 1){
	//	ez = 1;
	//}
	//if (ez > dz - 2){
	//	ez = dz - 2;
	//}


	if ((ex != 0) && (ex != (dx - 1)) && (ey != 0) && (ey != (dy - 1)) && (ez != 0) && (ez != (dz - 1))){
	
		int offset = pitch / sizeof(float4);

	
		float4 p0 = b[ez*offset + ey*NX + ex];//value b
	
	
		float4 p1 = temp[ez*offset + ey*NX + ex - 1];//left x
	
		float4 p2 = temp[ez*offset + ey*NX + ex + 1];//right x
					
		float4 p3 = temp[ez*offset + (ey - 1)*NX + ex];//top x
					
		float4 p4 = temp[ez*offset + (ey + 1)*NX + ex];//bottom x
					
		float4 p5 = temp[(ez - 1)*offset + ey*NX + ex];//front x
					
		float4 p6 = temp[(ez + 1)*offset + ey*NX + ex];//behind x

					
		v[ez*offset + ey*NX + ex] = rBeta * (p1 + p2 + p3 + p4 + p5 + p6 + alpha * p0);
	}
	__syncthreads();
}

__global__ void
divergence_k(float4 *d, float4 *v,
int dx, int dy, int dz, int lb, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16


	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;
	
	if (ex != 0 && ex != (dx - 1) && ey != 0 && ey != (dy - 1) && ez != 0 && ez != (dz - 1)){

		int offset = pitch / sizeof(float4);

		float4 p1 = v[ez*offset + ey*NX + ex - 1];//left x
	
		float4 p2 = v[ez*offset + ey*NX + ex + 1];//right x
	
		float4 p3 = v[ez*offset + (ey - 1)*NX + ex];//top x
	
		float4 p4 = v[ez*offset + (ey + 1)*NX + ex];//bottom x
	
		float4 p5 = v[(ez - 1)*offset + ey*NX + ex];//front x
	
		float4 p6 = v[(ez + 1)*offset + ey*NX + ex];//behind x

		float div = 0.5*((p2.x - p1.x) + (p4.y - p3.y) + (p6.z - p5.z));
		d[ez*offset + ey*NX + ex].x = div;
		d[ez*offset + ey*NX + ex].y = div;
		d[ez*offset + ey*NX + ex].z = div;
		d[ez*offset + ey*NX + ex].w = div;
	}
	else{
	//	boundary_condition_k(d, dx, dy, dz, ex, ey, ez, 1, pitch);
	}
	__syncthreads();
}



__global__ void
gradient_k(float4 *v, float4 *p,
int dx, int dy, int dz, int lb, size_t pitch)
{
	//dx = 64
	//dy = 64
	//dz = 16

	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;


	/*if (ex < 1){
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
	}*/
	if (ex != 0 && ex != (dx - 1) && ey != 0 && ey != (dy - 1) && ez != 0 && ez != (dz - 1)){

		int offset = pitch / sizeof(float4);
		float4 p1 = p[ez*offset + ey*NX + ex - 1];
			//right x
		float4 p2 = p[ez*offset + ey*NX + ex + 1];
			//top x
		float4 p3 = p[ez*offset + (ey - 1)*NX + ex];
			//bottom x
		float4 p4 = p[ez*offset + (ey + 1)*NX + ex];
			//front x
		float4 p5 = p[(ez - 1)*offset + ey*NX + ex];
			//behind x
		float4 p6 = p[(ez + 1)*offset + ey*NX + ex];


		float4 vel = v[ez*offset + ey*NX + ex];
		float4 grad;
		grad.x = 0.5*(p2.x - p1.x);
		grad.y = 0.5*(p4.x - p3.x);
		grad.z = 0.5*(p6.x - p5.x);
		
		v[ez*offset + ey*NX + ex] = vel - grad;
	}
	else{
	//	boundary_condition_k(p, dx, dy, dz, ex, ey, ez, 1, pitch);
	}
	__syncthreads();

	
	
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

	
	int index = ez*dx*dy + ey*dx + ex;


	float3 position = particle[index];

	/*float4 *vloc = (float4 *)
		((char *)v + ez * pitch) + ey * dy + ex;*/
	float3 newPosition;

	float4 vloc = tex3D(texref, position.x * dx, position.y * dy, position.z * dz);

	newPosition.x = position.x + dt * vloc.x;
	newPosition.y = position.y + dt * vloc.y;
	newPosition.z = position.z + dt * vloc.z;



	/*newPosition.x = (float)ex / dx;
	newPosition.y = (float)ey / dy;
	newPosition.z = (float)ez / dz;*/
	particle[index] = newPosition;
	__syncthreads();
	
}

__global__ void
bc_k(float4 *b, size_t pitch,float scale){
	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;
	if (ex != 0 && ex != (NX - 1) && ey != 0 && ey != (NY - 1) && ez != 0 && ez != (NZ - 1)){
		return;
	}
	else{
		boundary_condition_k(b, ex, ey, ez, scale, pitch);
	}
	__syncthreads();
}

__global__ void
force_k(float4 *v, int dx, int dy, int dz, float dt, size_t pitch){
	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;
	if (ex != 0 && ex != (dx - 1) && ey != 0 && ey != (dy - 1) && ez != 0 && ez != (dz - 1)){
		if (ey > 20){

			int offset = pitch / sizeof(float4);
			v[ez*offset + ey*NX + ex] = v[ez*offset + ey*NX + ex] - dt * make_float4(0, 0.009, 0, 0);
		}
	}
	else{
		
	}

}


extern "C"
void advect(float4 *v, float4 *temp, int dx, int dy, int dz, float dt)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);

	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);

	updateTexture(v, NX, NY, tPitch_v);
	advect_k<<<block_size, threads_size >>>(v, dx, dy, dz, dt, NY / THREAD_Y, tPitch_v);
	bc_k << <block_size, threads_size >> >(v, tPitch_v, -1.f);
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
		//xNew, x, b, alpha, rBeta, dx, dy, dz, pitch;
		jacobi_k << <block_size, threads_size >> >(v, temp, temp, alpha, rBeta, dx, dy, dz, tPitch_v);
		SWAP(v, temp);
	
	}
	
//	force_k << <block_size, threads_size >> >(v, dt, tPitch_v);

	getLastCudaError("diffuse_k failed.");
}

extern "C"
void projection(float4 *v, float4 *temp, float4 *pressure, float4* divergence, int dx, int dy, int dz, float dt)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);

	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	
	
	divergence_k<<<block_size, threads_size >>>(divergence, v, dx, dy, dz, NY / THREAD_Y, tPitch_v);
	bc_k << <block_size, threads_size >> >(divergence, tPitch_p, 1.f);
	cudaMemset(pressure, 0, sizeof(float4)*NX*NY*NZ);
	
	for(int i = 0; i < 40; i++){
		jacobi_k<<<block_size, threads_size >>>(temp, pressure, divergence, -1, 1.f / 6, dx, dy, dz, tPitch_v);
		SWAP(pressure, temp);
	}

	bc_k << <block_size, threads_size >> >(pressure, tPitch_p, 1.f);

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


	float4 *p1;
	cudaGraphicsMapResources(1, &cuda_vbo_resource1, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes1;
	cudaGraphicsResourceGetMappedPointer((void **)&p1, &num_bytes1, cuda_vbo_resource1);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

//	cudaMemset(p1, 60, num_bytes1);
	cudaMemcpy(p1, v, sizeof(float4) * DS, cudaMemcpyDeviceToDevice);

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource1, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");

}