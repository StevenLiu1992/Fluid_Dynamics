#include "fluidKernel.cuh"

texture<float4, 3, cudaReadModeElementType> texref_vel;
texture<float, 3, cudaReadModeElementType> texref_den;
texture<float, 3, cudaReadModeElementType> texref_levelset;
texture<float4, 3, cudaReadModeElementType> texref_temp;

static cudaArray *array_vel = NULL;
static cudaArray *array_temp = NULL;
static cudaArray *array_den = NULL;
static cudaArray *array_levelset = NULL;

cudaChannelFormatDesc ca_descriptor_1f;
cudaChannelFormatDesc ca_descriptor_4f;

cudaExtent volumeSize;



// Texture pitch
extern size_t tPitch_v;
extern size_t tPitch_t;
extern size_t tPitch_p;
extern size_t tPitch_d;
extern size_t tPitch_den;
extern size_t tPitch_lsf;

// Particle data
extern GLuint vbo;                 // OpenGL vertex buffer object
extern GLuint vbo2;                 // OpenGL vertex buffer object
extern GLuint vbo3;                 // OpenGL vertex buffer object
extern struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
extern struct cudaGraphicsResource *cuda_vbo_resource1; // handles OpenGL-CUDA exchange
extern struct cudaGraphicsResource *cuda_vbo_resource2; // handles OpenGL-CUDA exchange

void setupTexture(){
	
	volumeSize = make_cudaExtent(NX, NY, NZ);
	ca_descriptor_4f = cudaCreateChannelDesc<float4>();
	ca_descriptor_1f = cudaCreateChannelDesc<float>();

	//levelset texture
	texref_levelset.filterMode = cudaFilterModeLinear;
	texref_levelset.addressMode[0] = cudaAddressModeClamp;
	texref_levelset.addressMode[1] = cudaAddressModeClamp;
	texref_levelset.addressMode[2] = cudaAddressModeClamp;
	texref_levelset.normalized = false;

	getLastCudaError("cudaMalloc failed");
	checkCudaErrors(cudaMalloc3DArray(&array_levelset, &ca_descriptor_1f, volumeSize));

	//density texture
	texref_den.filterMode = cudaFilterModeLinear;
	texref_den.addressMode[0] = cudaAddressModeClamp;
	texref_den.addressMode[1] = cudaAddressModeClamp;
	texref_den.addressMode[2] = cudaAddressModeClamp;
	texref_den.normalized = false;

	getLastCudaError("cudaMalloc failed");
	checkCudaErrors(cudaMalloc3DArray(&array_den, &ca_descriptor_1f, volumeSize));

	//velocity texture
	texref_vel.filterMode = cudaFilterModeLinear;
	texref_vel.addressMode[0] = cudaAddressModeClamp;
	texref_vel.addressMode[1] = cudaAddressModeClamp;
	texref_vel.addressMode[2] = cudaAddressModeClamp;
	texref_vel.normalized = false;

	checkCudaErrors(cudaMalloc3DArray(&array_vel, &ca_descriptor_4f, volumeSize));
	getLastCudaError("cudaMalloc failed");
	
	//temp texture
	texref_temp.filterMode = cudaFilterModeLinear;
	texref_temp.addressMode[0] = cudaAddressModeClamp;
	texref_temp.addressMode[1] = cudaAddressModeClamp;
	texref_temp.addressMode[2] = cudaAddressModeClamp;

	checkCudaErrors(cudaMalloc3DArray(&array_temp, &ca_descriptor_4f, volumeSize));
	getLastCudaError("cudaMalloc failed");

}

void bindTexture(void){
	cudaBindTextureToArray(texref_vel, array_vel);
	cudaBindTextureToArray(texref_levelset, array_levelset);
	cudaBindTextureToArray(texref_den, array_den);
	getLastCudaError("cudaBindTexture failed");
}

void unbindTexture(void){
	cudaUnbindTexture(texref_vel);
	cudaUnbindTexture(texref_den);
}

void update_temp_texture(float4 *data,int dimx, int dimy, size_t pitch){
	cudaMemcpy3DParms cpy_params = { 0 };
	cpy_params.extent = volumeSize;
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.dstArray = array_temp;
	cpy_params.srcPtr = make_cudaPitchedPtr((void*)data, dimx*sizeof(float4), dimx, dimy);
	checkCudaErrors(cudaMemcpy3D(&cpy_params));
	getLastCudaError("cudaMemcpy failed");

}

void update_vel_texture(float4 *data, int dimx, int dimy, size_t pitch){
	cudaMemcpy3DParms cpy_params = { 0 };
	cpy_params.extent = volumeSize;
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.dstArray = array_vel;
	cpy_params.srcPtr = make_cudaPitchedPtr((void*)data, dimx*sizeof(float4), dimx, dimy);
	checkCudaErrors(cudaMemcpy3D(&cpy_params));
	getLastCudaError("cudaMemcpy failed");

}

void update_1f_texture(cudaArray *array_1d, float *data, int dimx, int dimy, size_t pitch){
	cudaMemcpy3DParms cpy_params = { 0 };
	cpy_params.extent = volumeSize;
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.dstArray = array_1d;
	cpy_params.srcPtr = make_cudaPitchedPtr((void*)data, dimx*sizeof(float), dimx, dimy);
	checkCudaErrors(cudaMemcpy3D(&cpy_params));
	getLastCudaError("cudaMemcpy failed");
}

__device__ float4 operator+(const float4 &a, const float4 &b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,a.w+b.w);
}
__device__ float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 operator-(const float3 &a, const float3 &b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float4 operator-(const float4 &a, const float4 &b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ float4 operator*(const float &a, const float4 &b) {
	return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__device__ void
boundary_density_condition_k(float *v, int ex, int ey, int ez, int scale, size_t pitch){
	
	int pitch0 = pitch / sizeof(float);
	
	//surface>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex == 0){
		//	float3 offset = make_float3(1, 0, 0);
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + ey*NX + ex + 1];
	}
	if (ex == (NX - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + ey*NX + ex - 1];
	}
	if (ey == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey + 1)*NX + ex];
	}
	if (ey == (NY - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey - 1)*NX + ex];
	}
	if (ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + ey*NX + ex];
	}
	if (ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + ey*NX + ex];
	}

	//edge>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//<<<<<<<<<<<<<<<<<<bottom four>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex != 0 && ex != (NX - 1) && ey == 0 && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey + 1)*NX + ex];
	}
	if (ex == (NX - 1) && ey == 0 && ez != 0 && ez != (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey + 1)*NX + ex - 1];
	}
	if (ex != 0 && ex != (NX - 1) && ey == 0 && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey + 1)*NX + ex];
	}
	if (ex == 0 && ey == 0 && ez != 0 && ez != (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey + 1)*NX + ex + 1];
	}

	//<<<<<<<<<<<<<<<<<<middle four>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex == 0 && ey != 0 && ey != (NY - 1) && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + ey*NX + ex + 1];
	}
	if (ex == (NX - 1) && ey != 0 && ey != (NY - 1) && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + ey*NX + ex - 1];
	}
	if (ex == (NX - 1) && ey != 0 && ey != (NY - 1) && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + ey*NX + ex - 1];
	}
	if (ex == 0 && ey != 0 && ey != (NY - 1) && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + ey*NX + ex + 1];
	}

	//<<<<<<<<<<<<<<<<<<top four>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex != 0 && ex != (NX - 1) && ey == (NY - 1) && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey - 1)*NX + ex];
	}
	if (ex == (NX - 1) && ey == (NY - 1) && ez != 0 && ez != (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey - 1)*NX + ex - 1];
	}
	if (ex != 0 && ex != (NX - 1) && ey == (NY - 1) && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey - 1)*NX + ex];
	}
	if (ex == 0 && ey == (NY - 1) && ez != 0 && ez != (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey - 1)*NX + ex + 1];
	}


	//corner>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex == 0 && ey == 0 && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey + 1)*NX + ex + 1];
	}
	if (ex == (NX - 1) && ey == 0 && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey + 1)*NX + ex - 1];
	}
	if (ex == 0 && ey == (NY - 1) && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey - 1)*NX + ex + 1];
	}
	if (ex == 0 && ey == 0 && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey + 1)*NX + ex + 1];
	}

	if (ex == (NX - 1) && ey == (NY - 1) && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey - 1)*NX + ex - 1];
	}
	if (ex == (NX - 1) && ey == 0 && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey + 1)*NX + ex - 1];
	}
	if (ex == 0 && ey == (NY - 1) && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey - 1)*NX + ex + 1];
	}
	if (ex == (NX - 1) && ey == (NY - 1) && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey - 1)*NX + ex - 1];
	}
	__syncthreads();
}

__device__ void
boundary_condition_k(float4 *v, int ex, int ey, int ez, int scale, size_t pitch){
	int pitch0 = pitch / sizeof(float4);
	
	//surface>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + ey*NX + ex + 1];
	}
	if (ex == (NX - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + ey*NX + ex - 1];
	}
	
	if (ey == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey + 1)*NX + ex];
	}
	if (ey == (NY - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey - 1)*NX + ex];
	}
	if (ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + ey*NX + ex];
	}
	if (ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + ey*NX + ex];
	}

	//edge>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//<<<<<<<<<<<<<<<<<<bottom four>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex != 0 && ex != (NX - 1) && ey == 0 && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey + 1)*NX + ex];
	}
	if (ex == (NX - 1) && ey == 0 && ez != 0 && ez != (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey + 1)*NX + ex - 1];
	}
	if (ex != 0 && ex != (NX - 1) && ey == 0 && ez == (NZ-1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey + 1)*NX + ex];
	}
	if (ex == 0 && ey == 0 && ez != 0 && ez != (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey + 1)*NX + ex + 1];
	}

	//<<<<<<<<<<<<<<<<<<middle four>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex == 0 && ey != 0 && ey != (NY - 1) && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + ey*NX + ex + 1];
	}
	if (ex == (NX - 1) && ey != 0 && ey != (NY - 1) && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + ey*NX + ex - 1];
	}
	if (ex == (NX - 1) && ey != 0 && ey != (NY - 1) && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + ey*NX + ex - 1];
	}
	if (ex == 0 && ey != 0 && ey != (NY - 1) && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + ey*NX + ex + 1];
	}

	//<<<<<<<<<<<<<<<<<<top four>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex != 0 && ex != (NX - 1) && ey == (NY - 1) && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey - 1)*NX + ex];
	}
	if (ex == (NX - 1) && ey == (NY - 1) && ez != 0 && ez != (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey - 1)*NX + ex - 1];
	}
	if (ex != 0 && ex != (NX - 1) && ey == (NY - 1) && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey - 1)*NX + ex];
	}
	if (ex == 0 && ey == (NY - 1) && ez != 0 && ez != (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[ez*pitch0 + (ey - 1)*NX + ex + 1];
	}


	//corner>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex == 0 && ey == 0 && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey + 1)*NX + ex + 1];
	}
	if (ex == (NX - 1) && ey == 0 && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey + 1)*NX + ex - 1];
	}
	if (ex == 0 && ey == (NY - 1) && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey - 1)*NX + ex + 1];
	}
	if (ex == 0 && ey == 0 && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey + 1)*NX + ex + 1];
	}

	if (ex == (NX - 1) && ey == (NY - 1) && ez == 0){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez + 1)*pitch0 + (ey - 1)*NX + ex - 1];
	}
	if (ex == (NX - 1) && ey == 0 && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey + 1)*NX + ex - 1];
	}
	if (ex == 0 && ey == (NY - 1) && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey - 1)*NX + ex + 1];
	}
	if (ex == (NX - 1) && ey == (NY - 1) && ez == (NZ - 1)){
		v[ez*pitch0 + ey*NX + ex] = scale * v[(ez - 1)*pitch0 + (ey - 1)*NX + ex - 1];
	}
	__syncthreads();
}


__global__ void
advect_k(float4 *v, int dx, int dy, int dz, float dt, size_t pitch)
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
		velocity = tex3D(texref_vel, ex + 0.5, ey + 0.5, ez + 0.5);
		ploc.x = (ex ) - dt * velocity.x * dx;
		ploc.y = (ey ) - dt * velocity.y * dy;
		ploc.z = (ez ) - dt * velocity.z * dz;





		velocity = tex3D(texref_vel, ploc.x + 0.5, ploc.y + 0.5, ploc.z + 0.5);

		float4 *Velocity_field = (float4 *)((char *)v + ez * pitch) + ey * dy + ex;
		(*Velocity_field) = velocity;	
	}

	__syncthreads();
}


__global__ void
advect_density_k(float *d, int dx, int dy, int dz, float dt, size_t pitch)
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

	if ((ex != 0) && (ex != (dx - 1)) && (ey != 0) && (ey != (dy - 1)) && (ez != 0) && (ez != (dz - 1))){

		float4 velocity;
		float3 ploc;
		float den;

		//	find the velocity of this position
		velocity = tex3D(texref_vel, ex + 0.5, ey + 0.5, ez + 0.5);
		
		//tracing back
		ploc.x = (ex) - dt * velocity.x * dx;
		ploc.y = (ey) - dt * velocity.y * dy;
		ploc.z = (ez) - dt * velocity.z * dz;

		//get the density of tracing back position
		den = tex3D(texref_den, ploc.x + 0.5, ploc.y + 0.5, ploc.z + 0.5);

	//	float *density = (float*)((char *)d + ez * pitch) + ey * dy + ex;
		d[ez*NX*NY + ey*NX + ex] = den;
		/*if (ey == 1)
			d[ez*NX*NY + ey*NX + ex] = velocity.y;*/
	//	(*density) = den;
	}

	__syncthreads();
}

__global__ void
advect_levelset_k(float *ls, int dx, int dy, int dz, float dt, size_t pitch){
	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	float4 velocity;
	float3 ploc;
	float new_levelset;

	//	find the velocity of this position
	velocity = tex3D(texref_vel, ex + 0.5, ey + 0.5, ez + 0.5);

	//tracing back
	ploc.x = (ex)-dt * velocity.x * dx;
	ploc.y = (ey)-dt * velocity.y * dy;
	ploc.z = (ez)-dt * velocity.z * dz;

	//get the density of tracing back position
	new_levelset = tex3D(texref_levelset, ploc.x + 0.5, ploc.y + 0.5, ploc.z + 0.5);
	ls[ez*NX*NY + ey*NX + ex] = new_levelset;

	__syncthreads();
}




__global__ void
jacobi_k(float4 *v, float4 *temp, float4 *b, float alpha, float rBeta,
int dx, int dy, int dz, size_t pitch){

	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;


	int left, right, top, bottom, front, behind;
	int offset = pitch / sizeof(float4);
	
	/*if (ex == 0)
		left = ez*offset + ey*NX + ex;
	else
		left = ez*offset + ey*NX + ex - 1;
	if (ex == NX - 1)
		right = ez*offset + ey*NX + ex;
	else
		right = ez*offset + ey*NX + ex + 1;
	if (ey == 0)
		bottom = ez*offset + ey*NX + ex;
	else
		bottom = ez*offset + (ey - 1)*NX + ex;
	if (ey == NY - 1)
		top = ez*offset + ey*NX + ex;
	else
		top = ez*offset + (ey + 1)*NX + ex;
	if (ez == 0)
		behind = ez*offset + ey*NX + ex;
	else
		behind = (ez - 1)*offset + ey*NX + ex;
	if (ex == NZ - 1)
		front = ez*offset + ey*NX + ex;
	else
		front = (ez + 1)*offset + ey*NX + ex;*/
	
	if (ex != 0 && ex != (dx - 1) && ey != 0 && ey != (dy - 1) && ez != 0 && ez != (dz - 1)){
		left =		ez*offset + ey*NX + ex - 1;
		right =		ez*offset + ey*NX + ex + 1;
		bottom =	ez*offset + (ey - 1)*NX + ex;
		top =		ez*offset + (ey + 1)*NX + ex;
		behind =	(ez - 1)*offset + ey*NX + ex;
		front =		(ez + 1)*offset + ey*NX + ex;
		float4 p1 = temp[left];//left x
		float4 p2 = temp[right];//right x
		float4 p3 = temp[bottom];//bottom x
		float4 p4 = temp[top];//top x
		float4 p6 = temp[front];//behind x
		float4 p5 = temp[behind];//front x

		float4 p0 = b[ez*offset + ey*NX + ex];//value b


		//float4 p1 = tex3D(texref_vel, ex - 1, ey, ez);
		//float4 p2 = tex3D(texref_vel, ex + 1, ey, ez);
		//float4 p3 = tex3D(texref_vel, ex, ey - 1, ez);
		//float4 p4 = tex3D(texref_vel, ex, ey + 1, ez);
		//float4 p5 = tex3D(texref_vel, ex, ey, ez - 1);
		//float4 p6 = tex3D(texref_vel, ex, ey, ez + 1);
		//float4 p0 = b[ez*offset + ey*NX + ex];//value b

		v[ez*offset + ey*NX + ex] = rBeta * (p1 + p2 + p3 + p4 + p5 + p6 + alpha * p0);
	}
	__syncthreads();
}

__global__ void
divergence_k(float4 *d, float4 *v,
int dx, int dy, int dz, int lb, size_t pitch)
{

	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;
	
	if (ex != 0 && ex != (dx - 1) && ey != 0 && ey != (dy - 1) && ez != 0 && ez != (dz - 1)){
		int left, right, top, bottom, front, behind;
		int offset = pitch / sizeof(float4);
			
		left	= ez*offset + ey*NX + ex - 1;
		right	= ez*offset + ey*NX + ex + 1;
		bottom	= ez*offset + (ey - 1)*NX + ex;
		top		= ez*offset + (ey + 1)*NX + ex;
		behind	= (ez - 1)*offset + ey*NX + ex;
		front	= (ez + 1)*offset + ey*NX + ex;
		float4 p1 = v[left];//left x
		float4 p2 = v[right];//right x
		float4 p3 = v[bottom];//bottom x
		float4 p4 = v[top];//top x
		float4 p5 = v[behind];//front x
		float4 p6 = v[front];//behind x

		float div = 0.5*((p2.x - p1.x) + (p4.y - p3.y) + (p6.z - p5.z));
		d[ez*offset + ey*NX + ex].x = div;
		d[ez*offset + ey*NX + ex].y = div;
		d[ez*offset + ey*NX + ex].z = div;
		d[ez*offset + ey*NX + ex].w = div;
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

	if (ex != 0 && ex != (dx - 1) && ey != 0 && ey != (dy - 1) && ez != 0 && ez != (dz - 1)){
		int left, right, top, bottom, front, behind;
		int offset = pitch / sizeof(float4);
		/*if (ex == 0)
			left = ez*offset + ey*NX + ex;
			else
			left = ez*offset + ey*NX + ex - 1;
			if (ex == NX - 1)
			right = ez*offset + ey*NX + ex;
			else
			right = ez*offset + ey*NX + ex + 1;
			if (ey == 0)
			bottom = ez*offset + ey*NX + ex;
			else
			bottom = ez*offset + (ey - 1)*NX + ex;
			if (ey == NY - 1)
			top = ez*offset + ey*NX + ex;
			else
			top = ez*offset + (ey + 1)*NX + ex;
			if (ez == 0)
			behind = ez*offset + ey*NX + ex;
			else
			behind = (ez - 1)*offset + ey*NX + ex;
			if (ex == NZ - 1)
			front = ez*offset + ey*NX + ex;
			else
			front = (ez + 1)*offset + ey*NX + ex;*/

		left =		ez*offset + ey*NX + ex - 1;
		right =		ez*offset + ey*NX + ex + 1;
		bottom =	ez*offset + (ey - 1)*NX + ex;
		top	=		ez*offset + (ey + 1)*NX + ex;
		behind =	(ez - 1)*offset + ey*NX + ex;
		front =		(ez + 1)*offset + ey*NX + ex;

		float4 p1 = p[left];//left x
		float4 p2 = p[right];//right x
		float4 p3 = p[bottom];//bottom x
		float4 p4 = p[top];//top x
		float4 p5 = p[behind];//front x
		float4 p6 = p[front];//behind x

		float4 vel = v[ez*offset + ey*NX + ex];
		float4 grad;
		grad.x = 0.5*(p2.x - p1.x);
		grad.y = 0.5*(p4.y - p3.y);
		grad.z = 0.5*(p6.z - p5.z);
		grad.w = 0;
		v[ez*offset + ey*NX + ex] = vel - grad;
		v[ez*offset + ey*NX + ex].w = 0;

		__syncthreads();
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

	float4 vloc = tex3D(texref_vel, position.x * dx+0.5, position.y * dy+0.5, position.z * dz+0.5);

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
advectParticles_Runge_Kutta_k(float3 *particle, float4 *v,
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
	float3 midPosition;

	float4 vloc = tex3D(texref_vel, position.x * dx + 0.5, position.y * dy + 0.5, position.z * dz + 0.5);

	//middle postion
	midPosition.x = position.x  + 0.5 * dt * vloc.x;
	midPosition.y = position.y  + 0.5 * dt * vloc.y;
	midPosition.z = position.z  + 0.5 * dt * vloc.z;
	float4 midVelocity = tex3D(texref_vel, midPosition.x*dx + 0.5, midPosition.y*dy + 0.5, midPosition.z*dz + 0.5);

	newPosition.x = (position.x + dt * midVelocity.x);
	newPosition.y = (position.y + dt * midVelocity.y);
	newPosition.z = (position.z + dt * midVelocity.z);

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

	if (ex == 0 || ex == (NX - 1) || ey == 0 || ey == (NY - 1) || ez == 0 || ez == (NZ - 1)){
		boundary_condition_k(b, ex, ey, ez, scale, pitch);
	}
	__syncthreads();
}

__global__ void
bc_density_k(float *b, size_t pitch, float scale){
	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;

	if (ex == 0 || ex == (NX - 1) || ey == 0 || ey == (NY - 1) || ez == 0 || ez == (NZ - 1)){
		boundary_density_condition_k(b, ex, ey, ez, scale, pitch);
	}
	__syncthreads();
}

__global__ void
force_k(float4 *v, float *d, float dt, size_t pitch){
	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;
	if (ex != 0 && ex != (NX - 1) && ey != 0 && ey != (NY - 1) && ez != 0 && ez != (NZ - 1)){
		int offset = pitch / sizeof(float4);
		if (d[ez*NX*NY + ey*NX + ex]>0.1)
			v[ez*offset + ey*NX + ex] = v[ez*offset + ey*NX + ex] - dt * make_float4(0, 0.009, 0, 0);
	}
}


extern "C"
void advect(float4 *v, int dx, int dy, int dz, float dt)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);

	update_vel_texture(v, NX, NY, tPitch_v);
	advect_k<<<block_size, threads_size >>>(v, dx, dy, dz, dt, tPitch_v);
	bc_k << <block_size, threads_size >> >(v, tPitch_v, -1.f);
	getLastCudaError("advectVelocity_k failed.");
}

extern "C"
void addForce(float4 *v, float *d, int dx, int dy, int dz, float dt){
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	force_k << <block_size, threads_size >> >(v, d, dt, tPitch_v);
	getLastCudaError("addForce failed.");
}

extern "C"
void advectDensity(float4 *v, float *d, int dx, int dy, int dz, float dt){
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	update_vel_texture(v, NX, NY, tPitch_v);
	update_1f_texture(array_den, d, NX, NY, tPitch_den);
	advect_density_k << <block_size, threads_size >> >(d, dx, dy, dz, dt, tPitch_den);
	bc_density_k << <block_size, threads_size >> >(d, tPitch_den, 1.f);
	getLastCudaError("advectDensity_k failed.");
}

extern "C"
void advectLevelSet(float4 *v, float *ls, int dx, int dy, int dz, float dt){
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	update_vel_texture(v, NX, NY, tPitch_v);
	update_1f_texture(array_levelset, ls, NX, NY, tPitch_lsf);
	advect_levelset_k << <block_size, threads_size >> >(ls, dx, dy, dz, dt, tPitch_den);
	getLastCudaError("advectLevelSet_k failed.");
}

__global__ void
correctLevelset_first_k(float3 *p, float2 *con, size_t pitch){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	float e_cons = 2.71828;

	float3 particle_location = p[ez*NX*NY + ey*NX + ez];
	float3 grid_location0;
	grid_location0.x = floor(particle_location.x * NX);
	grid_location0.y = floor(particle_location.y * NY);
	grid_location0.z = floor(particle_location.z * NZ);
	float3 grid_location[26];
	//bottom 9
	grid_location[0] = make_float3(grid_location0.x - 1, grid_location0.y - 1, grid_location0.z - 1);
	grid_location[1] = make_float3(grid_location0.x,	 grid_location0.y - 1, grid_location0.z - 1);
	grid_location[2] = make_float3(grid_location0.x + 1, grid_location0.y - 1, grid_location0.z - 1);
	grid_location[3] = make_float3(grid_location0.x - 1, grid_location0.y - 1, grid_location0.z);
	grid_location[4] = make_float3(grid_location0.x,	 grid_location0.y - 1, grid_location0.z);
	grid_location[5] = make_float3(grid_location0.x + 1, grid_location0.y - 1, grid_location0.z);
	grid_location[6] = make_float3(grid_location0.x - 1, grid_location0.y - 1, grid_location0.z + 1);
	grid_location[7] = make_float3(grid_location0.x,	 grid_location0.y - 1, grid_location0.z + 1);
	grid_location[8] = make_float3(grid_location0.x + 1, grid_location0.y - 1, grid_location0.z + 1);

	//middle 8
	grid_location[9] = make_float3(grid_location0.x - 1, grid_location0.y, grid_location0.z - 1);
	grid_location[10]= make_float3(grid_location0.x,	 grid_location0.y, grid_location0.z - 1);
	grid_location[11]= make_float3(grid_location0.x + 1, grid_location0.y, grid_location0.z - 1);
	grid_location[12]= make_float3(grid_location0.x - 1, grid_location0.y, grid_location0.z);
	grid_location[13] = make_float3(grid_location0.x + 1, grid_location0.y, grid_location0.z);
	grid_location[14] = make_float3(grid_location0.x - 1, grid_location0.y, grid_location0.z + 1);
	grid_location[15] = make_float3(grid_location0.x,	  grid_location0.y, grid_location0.z + 1);
	grid_location[16] = make_float3(grid_location0.x + 1, grid_location0.y, grid_location0.z + 1);

	//top 9
	grid_location[17] = make_float3(grid_location0.x - 1, grid_location0.y + 1, grid_location0.z - 1);
	grid_location[18] = make_float3(grid_location0.x,	  grid_location0.y + 1, grid_location0.z - 1);
	grid_location[19] = make_float3(grid_location0.x + 1, grid_location0.y + 1, grid_location0.z - 1);
	grid_location[20] = make_float3(grid_location0.x - 1, grid_location0.y + 1, grid_location0.z);
	grid_location[21] = make_float3(grid_location0.x,	  grid_location0.y + 1, grid_location0.z);
	grid_location[22] = make_float3(grid_location0.x + 1, grid_location0.y + 1, grid_location0.z);
	grid_location[23] = make_float3(grid_location0.x - 1, grid_location0.y + 1, grid_location0.z + 1);
	grid_location[24] = make_float3(grid_location0.x,	  grid_location0.y + 1, grid_location0.z + 1);
	grid_location[25] = make_float3(grid_location0.x + 1, grid_location0.y + 1, grid_location0.z + 1);

	for (int i = 0; i < 26; i++){
		if (grid_location[i].x >= 0 && grid_location[i].x < NX &&
			grid_location[i].y >= 0 && grid_location[i].y < NY &&
			grid_location[i].z >= 0 && grid_location[i].z < NZ){
			float sq_dis = 
				(particle_location.x - grid_location[i].x)*(particle_location.x - grid_location[i].x) +
				(particle_location.y - grid_location[i].y)*(particle_location.y - grid_location[i].y) +
				(particle_location.z - grid_location[i].z)*(particle_location.z - grid_location[i].z);


		}
	}

}

extern "C"
void correctLevelSet(float *ls, float2 *con, int dx, int dy, int dz, float dt){
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);

	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	//get location of particles>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	float3 *particle;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&particle, &num_bytes, cuda_vbo_resource);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	update_1f_texture(array_levelset, ls, NX, NY, tPitch_lsf);
	correctLevelset_first_k << <block_size, threads_size >> >(particle, con, tPitch_lsf);
	getLastCudaError("advectParticles_k failed.");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");
}

extern "C"
void diffuse(float4 *v, float4 *temp, int dx, int dy, int dz, float dt)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);

	float rdx = 1.f;
	float alpha = rdx / VISC / dt;
	float rBeta = 1.f / (6 + alpha);
	cudaMemcpy(temp, v, sizeof(float4) * DS, cudaMemcpyDeviceToDevice);
	for(int i=0;i<20;i++){
		//xNew, x, b, alpha, rBeta, dx, dy, dz, pitch;
	//	update_vel_texture(temp, NX, NY, tPitch_v);
		jacobi_k << <block_size, threads_size >> >(v, temp, temp, alpha, rBeta, dx, dy, dz, tPitch_v);
		bc_k << <block_size, threads_size >> >(v, tPitch_v, -1.f);
		SWAP(v, temp);
	}
	getLastCudaError("diffuse_k failed.");
}

extern "C"
void projection(float4 *v, float4 *temp, float4 *pressure, float4* divergence, int dx, int dy, int dz, float dt)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);

	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	
	cudaMemset(divergence, 0, sizeof(float4)*NX*NY*NZ);
	divergence_k<<<block_size, threads_size >>>(divergence, v, dx, dy, dz, NY / THREAD_Y, tPitch_v);
	bc_k << <block_size, threads_size >> >(divergence, tPitch_p, 1.f);
	cudaMemset(pressure, 0, sizeof(float4)*NX*NY*NZ);
	
	for(int i = 0; i < 60; i++){
	//	update_vel_texture(pressure, NX, NY, tPitch_v);
		jacobi_k << <block_size, threads_size >> >(temp, pressure, divergence, -1, 1.f / 6, dx, dy, dz, tPitch_v);
		bc_k << <block_size, threads_size >> >(pressure, tPitch_p, 1.f);
		SWAP(pressure, temp);
	}

	

	gradient_k<<<block_size, threads_size >>>(v, pressure, dx, dy, dz, NY / THREAD_Y, tPitch_v);
	bc_k << <block_size, threads_size >> >(v, tPitch_v, -1.f);
	getLastCudaError("diffuse_k failed.");
}


extern "C"
void advectParticles(GLuint vbo, float4 *v, float *d, int dx, int dy, int dz, float dt)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);

	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	//change location of particles>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	float3 *p;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&p, &num_bytes, cuda_vbo_resource);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	update_vel_texture(v, NX, NY, tPitch_v);
	advectParticles_Runge_Kutta_k << <block_size, threads_size >> >(p, v, dx, dy, dz, dt, NY / THREAD_Y, tPitch_v);
	getLastCudaError("advectParticles_k failed.");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");

	//change velocity field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	float4 *p1;
	cudaGraphicsMapResources(1, &cuda_vbo_resource1, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes1;
	cudaGraphicsResourceGetMappedPointer((void **)&p1, &num_bytes1, cuda_vbo_resource1);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");


	cudaMemcpy(p1, v, sizeof(float4) * DS, cudaMemcpyDeviceToDevice);
	
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource1, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");


	//change density field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	float *p2;
	cudaGraphicsMapResources(1, &cuda_vbo_resource2, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes2;
	cudaGraphicsResourceGetMappedPointer((void **)&p2, &num_bytes2, cuda_vbo_resource2);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");


	cudaMemcpy(p2, d, sizeof(float)* DS, cudaMemcpyDeviceToDevice);	

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource2, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");
}
