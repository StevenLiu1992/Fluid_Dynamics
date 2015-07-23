#include "fluidKernel.cuh"

texture<float4, 3, cudaReadModeElementType> texref_vel;
texture<float, 3, cudaReadModeElementType> texref_den;
texture<float, 3, cudaReadModeElementType> texref_levelset;
texture<float4, 3, cudaReadModeElementType> texref_temp;
texture<float4, 2, cudaReadModeElementType> texref_ray;

extern int window_width;
extern int window_height;

static cudaArray *array_vel = NULL;
static cudaArray *array_temp = NULL;
static cudaArray *array_den = NULL;
static cudaArray *array_levelset = NULL;
static cudaArray *array_ray = NULL;

cudaChannelFormatDesc ca_descriptor_1f;
cudaChannelFormatDesc ca_descriptor_4f;

cudaExtent volumeSize;

extern float4 *hintersection;
extern float3 *hnormal;

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
extern struct cudaGraphicsResource *textureCudaResource; // handles OpenGL-CUDA exchange
extern struct cudaGraphicsResource *cuda_vbo_intersection; // handles OpenGL-CUDA exchange
extern struct cudaGraphicsResource *cuda_vbo_normal; // handles OpenGL-CUDA exchange





void setupTexture(){
	
	
	ca_descriptor_4f = cudaCreateChannelDesc<float4>();
	ca_descriptor_1f = cudaCreateChannelDesc<float>();

	//levelset texture
	volumeSize = make_cudaExtent(LNX, LNY, LNZ);
	texref_levelset.filterMode = cudaFilterModeLinear;
	texref_levelset.addressMode[0] = cudaAddressModeClamp;
	texref_levelset.addressMode[1] = cudaAddressModeClamp;
	texref_levelset.addressMode[2] = cudaAddressModeClamp;
	texref_levelset.normalized = false;

	checkCudaErrors(cudaMalloc3DArray(&array_levelset, &ca_descriptor_1f, volumeSize));
	getLastCudaError("cudaMalloc failed");

	volumeSize = make_cudaExtent(NX, NY, NZ);
	//density texture
	texref_den.filterMode = cudaFilterModeLinear;
	texref_den.addressMode[0] = cudaAddressModeClamp;
	texref_den.addressMode[1] = cudaAddressModeClamp;
	texref_den.addressMode[2] = cudaAddressModeClamp;
	texref_den.normalized = false;

	checkCudaErrors(cudaMalloc3DArray(&array_den, &ca_descriptor_1f, volumeSize));
	getLastCudaError("cudaMalloc failed");

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
	checkCudaErrors(cudaBindTextureToArray(&texref_vel, array_vel, &ca_descriptor_4f));
	cudaBindTextureToArray(&texref_levelset, array_levelset, &ca_descriptor_1f);
	cudaBindTextureToArray(&texref_den, array_den, &ca_descriptor_1f);
	getLastCudaError("cudaBindTexture failed");
}

void unbindTexture(void){
	cudaUnbindTexture(texref_vel);
	cudaUnbindTexture(texref_den);
	cudaUnbindTexture(texref_temp);
	cudaUnbindTexture(texref_levelset);
	cudaUnbindTexture(texref_ray);

	cudaFreeArray(array_temp);
	cudaFreeArray(array_levelset);
	cudaFreeArray(array_den);
	cudaFreeArray(array_vel);
}

void update_temp_texture(float4 *data,int dimx, int dimy, size_t pitch){
	cudaMemcpy3DParms cpy_params = { 0 };
	cpy_params.extent = make_cudaExtent(dimx, dimx, dimx);
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.dstArray = array_temp;
	cpy_params.srcPtr = make_cudaPitchedPtr((void*)data, dimx*sizeof(float4), dimx, dimy);
	checkCudaErrors(cudaMemcpy3D(&cpy_params));
	getLastCudaError("cudaMemcpy failed");

}

void update_vel_texture(float4 *data, int dimx, int dimy, size_t pitch){
	cudaMemcpy3DParms cpy_params = { 0 };
	
	cpy_params.extent = make_cudaExtent(dimx, dimx, dimx);
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.dstArray = array_vel;
	cpy_params.srcPtr = make_cudaPitchedPtr((void*)data, dimx*sizeof(float4), dimx, dimy);
	checkCudaErrors(cudaMemcpy3D(&cpy_params));
	getLastCudaError("cudaMemcpy failed");

}

void update_1f_texture(cudaArray *array_1d, float *data, int dimx, int dimy, size_t pitch){
	cudaMemcpy3DParms cpy_params = { 0 };
	cpy_params.extent = make_cudaExtent(dimx, dimx, dimx);
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.dstArray = array_1d;
	cpy_params.srcPtr = make_cudaPitchedPtr((void*)data, dimx*sizeof(float), dimx, dimy);
	checkCudaErrors(cudaMemcpy3D(&cpy_params));
	getLastCudaError("cudaMemcpy failed");
}

void bindTexturetoCudaArray(){
	checkCudaErrors(cudaGraphicsMapResources(1, &textureCudaResource, 0));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array_ray, textureCudaResource, 0, 0));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
	texref_ray.filterMode = cudaFilterModeLinear;
	texref_ray.addressMode[0] = cudaAddressModeClamp;
	texref_ray.addressMode[1] = cudaAddressModeClamp;

	checkCudaErrors(cudaBindTextureToArray(&texref_ray, array_ray, &ca_descriptor_4f));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
	cudaGraphicsUnmapResources(1, &textureCudaResource, 0);
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
__device__ float3 operator*(const float &a, const float3 &b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}
__device__ float operator*(const float3 &a, const float3 &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 operator/(const float3 &b, const float &a) {
	return make_float3(b.x/a, b.y/a, b.z/a);
}


__device__ float3 normalize(const float3 &a) {
	float3 dir = a;//the direction from camera to ray inter in model space
	dir = rsqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z) * dir;
	return dir;
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
boundary_levelset_condition_k(float *v, int ex, int ey, int ez, int scale, size_t pitch){

	int pitch0 = pitch / sizeof(float);

	//surface>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex == 0){
		//	float3 offset = make_float3(1, 0, 0);
		v[ez*pitch0 + ey*LNX + ex] = scale * v[ez*pitch0 + ey*LNX + ex + 1];
	}
	if (ex == (LNX - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[ez*pitch0 + ey*LNX + ex - 1];
	}
	if (ey == 0){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[ez*pitch0 + (ey + 1)*LNX + ex];
	}
	if (ey == (LNY - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[ez*pitch0 + (ey - 1)*LNX + ex];
	}
	if (ez == 0){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez + 1)*pitch0 + ey*LNX + ex];
	}
	if (ez == (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez - 1)*pitch0 + ey*LNX + ex];
	}

	//edge>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//<<<<<<<<<<<<<<<<<<bottom four>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex != 0 && ex != (LNX - 1) && ey == 0 && ez == 0){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez + 1)*pitch0 + (ey + 1)*LNX + ex];
	}
	if (ex == (LNX - 1) && ey == 0 && ez != 0 && ez != (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[ez*pitch0 + (ey + 1)*LNX + ex - 1];
	}
	if (ex != 0 && ex != (LNX - 1) && ey == 0 && ez == (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez - 1)*pitch0 + (ey + 1)*LNX + ex];
	}
	if (ex == 0 && ey == 0 && ez != 0 && ez != (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[ez*pitch0 + (ey + 1)*LNX + ex + 1];
	}

	//<<<<<<<<<<<<<<<<<<middle four>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex == 0 && ey != 0 && ey != (LNY - 1) && ez == 0){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez + 1)*pitch0 + ey*LNX + ex + 1];
	}
	if (ex == (LNX - 1) && ey != 0 && ey != (LNY - 1) && ez == 0){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez + 1)*pitch0 + ey*LNX + ex - 1];
	}
	if (ex == (LNX - 1) && ey != 0 && ey != (LNY - 1) && ez == (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez - 1)*pitch0 + ey*LNX + ex - 1];
	}
	if (ex == 0 && ey != 0 && ey != (LNY - 1) && ez == (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez - 1)*pitch0 + ey*LNX + ex + 1];
	}

	//<<<<<<<<<<<<<<<<<<top four>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex != 0 && ex != (LNX - 1) && ey == (LNY - 1) && ez == 0){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez + 1)*pitch0 + (ey - 1)*LNX + ex];
	}
	if (ex == (LNX - 1) && ey == (LNY - 1) && ez != 0 && ez != (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[ez*pitch0 + (ey - 1)*LNX + ex - 1];
	}
	if (ex != 0 && ex != (LNX - 1) && ey == (LNY - 1) && ez == (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez - 1)*pitch0 + (ey - 1)*LNX + ex];
	}
	if (ex == 0 && ey == (LNY - 1) && ez != 0 && ez != (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[ez*pitch0 + (ey - 1)*LNX + ex + 1];
	}


	//corner>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	if (ex == 0 && ey == 0 && ez == 0){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez + 1)*pitch0 + (ey + 1)*LNX + ex + 1];
	}
	if (ex == (LNX - 1) && ey == 0 && ez == 0){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez + 1)*pitch0 + (ey + 1)*LNX + ex - 1];
	}
	if (ex == 0 && ey == (LNY - 1) && ez == 0){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez + 1)*pitch0 + (ey - 1)*LNX + ex + 1];
	}
	if (ex == 0 && ey == 0 && ez == (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez - 1)*pitch0 + (ey + 1)*LNX + ex + 1];
	}

	if (ex == (LNX - 1) && ey == (LNY - 1) && ez == 0){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez + 1)*pitch0 + (ey - 1)*LNX + ex - 1];
	}
	if (ex == (LNX - 1) && ey == 0 && ez == (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez - 1)*pitch0 + (ey + 1)*LNX + ex - 1];
	}
	if (ex == 0 && ey == (LNY - 1) && ez == (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez - 1)*pitch0 + (ey - 1)*LNX + ex + 1];
	}
	if (ex == (LNX - 1) && ey == (LNY - 1) && ez == (LNZ - 1)){
		v[ez*pitch0 + ey*LNX + ex] = scale * v[(ez - 1)*pitch0 + (ey - 1)*LNX + ex - 1];
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
//	v[ez*pitch0 + ey*NX + ex].w = 1;
	__syncthreads();
}

__global__ void
bc_k(float4 *b, size_t pitch, float scale){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
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
bc_levelset_k(float *l, size_t pitch, float scale){
	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;

	if (ex == 0 || ex == (LNX - 1) || ey == 0 || ey == (LNY - 1) || ez == 0 || ez == (LNZ - 1)){
		boundary_levelset_condition_k(l, ex, ey, ez, scale, pitch);
	}
	__syncthreads();
}


///////////////////////////////////////////////////////////////////////////////////////////////
//welcome to navier stokes velocity equation><><><><><><><><><><><><><><><><><><><><><><><><><>
///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void
advect_k(float4 *v){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	if ((ex != 0) && (ex != (NX - 1)) && (ey != 0) && (ey != (NY - 1)) && (ez != 0) && (ez != (NZ - 1))){
		//semi-lagrangian method to advect velocity field
		float4 velocity;
		float3 ploc;
		float density;
		velocity = tex3D(texref_vel, ex + 0.5, ey + 0.5, ez + 0.5);
		//density = tex3D(texref_den, ex + 0.5, ey + 0.5, ez + 0.5);
		//if (density == 0){
		//	v[ez*NY*NZ + ey*NX + ex] = make_float4(0, 0, 0, 0);
		//	return;//in the air
		//}
		float ls = tex3D(texref_levelset, 2 * ex + 0.5, 2 * ey + 0.5, 2 * ez + 0.5);
		//if (ls > 0){
		////	v[ez*NY*NZ + ey*NX + ex] = make_float4(0,0,0,0);
		//	return;
		//}
		ploc.x = ex + 0.5 - DT * velocity.x * NX;
		ploc.y = ey + 0.5 - DT * velocity.y * NY;
		ploc.z = ez + 0.5 - DT * velocity.z * NZ;
//		density = tex3D(texref_den, ploc.x, ploc.y, ploc.z);
//		if (density == 0)
		v[ez*NY*NZ + ey*NX + ex] = tex3D(texref_vel, ploc.x, ploc.y, ploc.z);
	}

	__syncthreads();
}

__global__ void
jacobi_diffuse_k(float4 *v, float4 *temp, float4 *b, float *l, float alpha, float rBeta, size_t pitch){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	float th = 1;

	if (ex != 0 && ex != (NX - 1) && ey != 0 && ey != (NY - 1) && ez != 0 && ez != (NZ - 1)){
		if (l[2 * ez*LNX*LNY + 2 * ey*LNX + 2 * ex] > th &&
			l[2 * ez*LNX*LNY + 2 * ey*LNX + (2 * ex - 1)] > th &&
			l[2 * ez*LNX*LNY + 2 * ey*LNX + (2 * ex + 1)] > th &&
			l[2 * ez*LNX*LNY + (2 * ey - 1)*LNX + 2 * ex] > th &&
			l[2 * ez*LNX*LNY + (2 * ey + 1)*LNX + 2 * ex] > th &&
			l[(2 * ez - 1)*LNX*LNY + 2 * ey*LNX + 2 * ex] > th &&
			l[(2 * ez + 1)*LNX*LNY + 2 * ey*LNX + 2 * ex] > th)
			//outside of liquid
			return;

		int offset = pitch / sizeof(float4);

		int left = ez*offset + ey*NX + (ex - 1);
		int righ = ez*offset + ey*NX + (ex + 1);
		int bott = ez*offset + (ey - 1)*NX + ex;
		int topp = ez*offset + (ey + 1)*NX + ex;
		int back = (ez - 1)*offset + ey*NX + ex;
		int fron = (ez + 1)*offset + ey*NX + ex;

		float4 p1 = temp[left];
		float4 p2 = temp[righ];
		float4 p3 = temp[bott];
		float4 p4 = temp[topp];
		float4 p5 = temp[back];
		float4 p6 = temp[fron];


		float4 p0 = tex3D(texref_vel, ex + 0.5, ey + 0.5, ez + 0.5);

		v[ez*offset + ey*NX + ex] = rBeta * (p1 + p2 + p3 + p4 + p5 + p6 + alpha * p0);
	}
	__syncthreads();
}

__global__ void
jacobi_k(float4 *v, float4 *temp, float4 *b, float alpha, float rBeta, size_t pitch){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	if (ex != 0 && ex != (NX - 1) && ey != 0 && ey != (NY - 1) && ez != 0 && ez != (NZ - 1)){
		int offset = pitch / sizeof(float4);

		int left = ez*offset + ey*NX + (ex - 1);
		int righ = ez*offset + ey*NX + (ex + 1);
		int bott = ez*offset + (ey - 1)*NX + ex;
		int topp = ez*offset + (ey + 1)*NX + ex;
		int back = (ez - 1)*offset + ey*NX + ex;
		int fron = (ez + 1)*offset + ey*NX + ex;

		float4 p1 = temp[left];
		float4 p2 = temp[righ];
		float4 p3 = temp[bott];
		float4 p4 = temp[topp];
		float4 p5 = temp[back];
		float4 p6 = temp[fron];


		float4 p0 = tex3D(texref_vel, ex + 0.5, ey + 0.5, ez + 0.5);

		v[ez*offset + ey*NX + ex] = rBeta * (p1 + p2 + p3 + p4 + p5 + p6 + alpha * p0);
	}
	__syncthreads();
}

__global__ void
divergence_k(float4 *d, float4 *v, size_t pitch){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	if (ex != 0 && ex != (NX - 1) && ey != 0 && ey != (NY - 1) && ez != 0 && ez != (NZ - 1)){

		//divergence = 
		//u(i+1,j,k) - u(i-1,j,k) / 2dx + 
		//u(i,j+1,k) - u(i,j-1,k) / 2dy +
		//u(i,j,k+1) - u(i,j,k-1) / 2dz
		int offset = pitch / sizeof(float4);

		int left = ez*offset + ey*NX + (ex - 1);
		int righ = ez*offset + ey*NX + (ex + 1);
		int bott = ez*offset + (ey - 1)*NX + ex;
		int topp = ez*offset + (ey + 1)*NX + ex;
		int back = (ez - 1)*offset + ey*NX + ex;
		int fron = (ez + 1)*offset + ey*NX + ex;

		float4 p1 = v[left];
		float4 p2 = v[righ];
		float4 p3 = v[bott];
		float4 p4 = v[topp];
		float4 p5 = v[back];
		float4 p6 = v[fron];

		float div = 0.5*((p2.x - p1.x) + (p4.y - p3.y) + (p6.z - p5.z));
		d[ez*offset + ey*NX + ex].x = div;
		d[ez*offset + ey*NX + ex].y = div;
		d[ez*offset + ey*NX + ex].z = div;
		d[ez*offset + ey*NX + ex].w = div;
	}
	__syncthreads();
}

__global__ void
gradient_k(float4 *v, float4 *p, float *l, size_t pitch){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	float th = 0;
	
	if (ex != 0 && ex != (NX - 1) && ey != 0 && ey != (NY - 1) && ez != 0 && ez != (NZ - 1)){
		if (l[2 * ez*LNX*LNY + 2 * ey*LNX + 2 * ex] > th/* &&
			l[2 * ez*LNX*LNY + 2 * ey*LNX + (2 * ex - 1)] > th &&
			l[2 * ez*LNX*LNY + 2 * ey*LNX + (2 * ex + 1)] > th &&
			l[2 * ez*LNX*LNY + (2 * ey - 1)*LNX + 2 * ex] > th &&
			l[2 * ez*LNX*LNY + (2 * ey + 1)*LNX + 2 * ex] > th &&
			l[(2 * ez - 1)*LNX*LNY + 2 * ey*LNX + 2 * ex] > th &&
			l[(2 * ez + 1)*LNX*LNY + 2 * ey*LNX + 2 * ex] > th*/)
			//outside of liquid
			return;
		int offset = pitch / sizeof(float4);

		int left = ez*offset + ey*NX + (ex - 1);
		int righ = ez*offset + ey*NX + (ex + 1);
		int bott = ez*offset + (ey - 1)*NX + ex;
		int topp = ez*offset + (ey + 1)*NX + ex;
		int back = (ez - 1)*offset + ey*NX + ex;
		int fron = (ez + 1)*offset + ey*NX + ex;

		float4 p1 = p[left];
		float4 p2 = p[righ];
		float4 p3 = p[bott];
		float4 p4 = p[topp];
		float4 p5 = p[back];
		float4 p6 = p[fron];

		float4 vel = v[ez*offset + ey*NX + ex];
		float4 grad;
		grad.x = 0.5*(p2.x - p1.x);
		grad.y = 0.5*(p4.y - p3.y);
		grad.z = 0.5*(p6.z - p5.z);
		grad.w = 0;
		v[ez*offset + ey*NX + ex] = vel - grad;
		//	v[ez*offset + ey*NX + ex].w = 0;

		__syncthreads();
	}


}

__global__ void
force_k(float4 *v, float *l, size_t pitch){
	// ex is the domain location in x for this thread
	int ex = threadIdx.x + blockIdx.x * 8;
	// ey is the domain location in y for this thread
	int ey = threadIdx.y + blockIdx.y * 8;
	// ez is the domain location in z for this thread
	int ez = threadIdx.z + blockIdx.z * 8;
	if (ex != 0 && ex != (NX - 1) && ey != 0 && ey != (NY - 1) && ez != 0 && ez != (NZ - 1)){
		int offset = pitch / sizeof(float4);
		if (l[2*ez*LNX*LNY + 2*ey*LNX + 2* ex] <= 1){
			v[ez*offset + ey*NX + ex] = v[ez*offset + ey*NX + ex] - DT * make_float4(0, 0.009, 0, 0);
		//	v[ez*offset + ey*NX + ex].w = 1;
		}
		else{
		//	v[ez*offset + ey*NX + ex] = make_float4(0, 0, 0, 0);
		//	v[ez*offset + ey*NX + ex].w = 0;
		}
	}
}

__global__ void
exterapolation_k(float4 *v, float4 *temp, float *l){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;



	if (ex != 0 && ex != (NX - 1) && ey != 0 && ey != (NY - 1) && ez != 0 && ez != (NZ - 1)){
		float midd_levelset = l[2 * ez*LNX*LNY + 2 * ey*LNX + 2 * ex];
		if (midd_levelset <= 0)
			return;//in the liquid

		float left_levelset = l[2 * ez*LNX*LNY + 2 * ey*LNX + (2 * ex - 1)];
		float righ_levelset = l[2 * ez*LNX*LNY + 2 * ey*LNX + (2 * ex + 1)];
		float bott_levelset = l[2 * ez*LNX*LNY + (2 * ey - 1)*LNX + 2 * ex];
		float topp_levelset = l[2 * ez*LNX*LNY + (2 * ey + 1)*LNX + 2 * ex];
		float back_levelset = l[(2 * ez - 1)*LNX*LNY + 2 * ey*LNX + 2 * ex];
		float fron_levelset = l[(2 * ez + 1)*LNX*LNY + 2 * ey*LNX + 2 * ex];

		

		int s0, s1, s2;
		//x - direction
		if (midd_levelset > righ_levelset && midd_levelset > left_levelset){
			if (left_levelset > righ_levelset){
				s0 = 1;
			}
			else{
				s0 = -1;
			}
		}
		else if (midd_levelset > righ_levelset){
			s0 = 1;
		}
		else if (midd_levelset > left_levelset){
			s0 = -1;
		}
		else{
			s0 = 0;
		}

		//y - direction
		if (midd_levelset > topp_levelset && midd_levelset > bott_levelset){
			if (bott_levelset > topp_levelset){
				s1 = 1;
			}
			else{
				s1 = -1;
			}
		}
		else if (midd_levelset > topp_levelset){
			s1 = 1;
		}
		else if (midd_levelset > bott_levelset){
			s1 = -1;
		}
		else{
			s1 = 0;
		}

		//z - direction
		if (midd_levelset > fron_levelset && midd_levelset > back_levelset){
			if (back_levelset > fron_levelset){
				s2 = 1;
			}
			else{
				s2 = -1;
			}
		}
		else if (midd_levelset > fron_levelset){
			s2 = 1;
		}
		else if (midd_levelset > back_levelset){
			s2 = -1;
		}
		else{
			s2 = 0;
		}

		float3 fi0 = make_float3(midd_levelset, midd_levelset, midd_levelset);
		float3 fi1 = make_float3(
			l[2 * ez*LNX*LNY + 2 * ey*LNX + (2 * ex - s0)],
			l[2 * ez*LNX*LNY + (2 * ey - s1)*LNX + 2 * ex],
			l[(2 * ez - s2)*LNX*LNY + 2 * ey*LNX + 2 * ex]);
		float3 vx0 = make_float3(
			temp[ez*NX*NY + ey*NX + (ex - s0)].x,
			temp[ez*NX*NY + (ey - s1)*NX + ex].x,
			temp[(ez - s2)*NX*NY + ey*NX + ex].x
			);
		float3 vy0 = make_float3(
			temp[ez*NX*NY + ey*NX + (ex - s0)].y,
			temp[ez*NX*NY + (ey - s1)*NX + ex].y,
			temp[(ez - s2)*NX*NY + ey*NX + ex].y
			);
		float3 vz0 = make_float3(
			temp[ez*NX*NY + ey*NX + (ex - s0)].z,
			temp[ez*NX*NY + (ey - s1)*NX + ex].z,
			temp[(ez - s2)*NX*NY + ey*NX + ex].z
			);
		float3 I = make_float3(1, 1, 1);
		float3 fi2 = fi0 - fi1;
		float vx1 = (vx0*fi2) / (I*fi2);
		float vy1 = (vy0*fi2) / (I*fi2);
		float vz1 = (vz0*fi2) / (I*fi2);

		v[ez*NX*NY + ey*NX + ex] = make_float4(vx1, vy1, vz1, 1);
	
	}
	__syncthreads();
}

extern "C"
void exterapolation(float4 *v, float4 *temp, float *ls){
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	cudaMemcpy(temp, v, sizeof(float4) * DS, cudaMemcpyDeviceToDevice);
	for (int i = 0; i < 6; i++){

		exterapolation_k << <block_size, threads_size >> >(v, temp, ls);
		bc_k << <block_size, threads_size >> >(v, tPitch_v, -1.f);
		SWAP(v, temp);
	}
	getLastCudaError("exterapolation_Velocity_k failed.");
}

extern "C"
void advect(float4 *v, float *ls)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);

	update_vel_texture(v, NX, NY, tPitch_v);
	//update_1f_texture(array_den, d, NX, NY, tPitch_den);
	update_1f_texture(array_levelset, ls, LNX, LNY, tPitch_lsf);
	advect_k << <block_size, threads_size >> >(v);
	bc_k << <block_size, threads_size >> >(v, tPitch_v, -1.f);
	getLastCudaError("advectVelocity_k failed.");
}


extern "C"
void diffuse(float4 *v, float4 *temp, float *l)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);

	float rdx = 1.f;
	float alpha = rdx*rdx / VISC / DT;
	float rBeta = 1.f / (6 + alpha);
	cudaMemcpy(temp, v, sizeof(float4) * DS, cudaMemcpyDeviceToDevice);
	update_vel_texture(v, NX, NY, tPitch_v);//use for b
	for (int i = 0; i<20; i++){
		//xNew, x, b, alpha, rBeta, dx, dy, dz, pitch;
		jacobi_diffuse_k << <block_size, threads_size >> >(v, temp, temp, l, alpha, rBeta, tPitch_v);
		bc_k << <block_size, threads_size >> >(v, tPitch_v, -1.f);
		SWAP(v, temp);
	}
	getLastCudaError("diffuse_k failed.");
}

extern "C"
void projection(float4 *v, float4 *temp, float4 *pressure, float4* divergence, float *l)
{
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);

	cudaMemset(divergence, 0, sizeof(float4)*NX*NY*NZ);
	cudaMemset(temp, 0, sizeof(float4)*NX*NY*NZ);
	cudaMemset(pressure, 0, sizeof(float4)*NX*NY*NZ);

	divergence_k << <block_size, threads_size >> >(divergence, v, tPitch_v);
	bc_k << <block_size, threads_size >> >(divergence, tPitch_p, 1.f);
	
	update_vel_texture(divergence, NX, NY, tPitch_v);//use for b
	for (int i = 0; i < 60; i++){
		//	update_vel_texture(pressure, NX, NY, tPitch_v);
		jacobi_diffuse_k << <block_size, threads_size >> >(temp, pressure, divergence, l, -1.f, 1.f / 6, tPitch_v);
		bc_k << <block_size, threads_size >> >(temp, tPitch_p, 1.f);
		SWAP(pressure, temp);
	}
	gradient_k << <block_size, threads_size >> >(v, pressure, l, tPitch_v);
	bc_k << <block_size, threads_size >> >(v, tPitch_v, -1.f);
	getLastCudaError("diffuse_k failed.");
}

extern "C"
void addForce(float4 *v, float *l){
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	force_k << <block_size, threads_size >> >(v, l, tPitch_v);
	bc_k << <block_size, threads_size >> >(v, tPitch_v, -1.f);
	getLastCudaError("addForce failed.");
}
///////////////////////////////////////////////////////////////////////////////////////////////
//happy navier stokes velocity equation<><><><><><><><><><><><><><><><><><><><><><><><><><><><>
///////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////
//lets advect something(i.e. particles, density)<><><><><><><><><><><><><><><><><><><><><><><><
///////////////////////////////////////////////////////////////////////////////////////////////
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
advectParticles_Runge_Kutta_k(float3 *particle, float4 *v, size_t pitch){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	int index = ez*NX*NY + ey*NX + ex;
	float3 position = particle[index];

	float3 newPosition;
	float3 midPosition;

	float4 vloc = tex3D(texref_vel, position.x * NX + 0.5, position.y * NY + 0.5, position.z * NZ + 0.5);

	//middle postion
	midPosition.x = position.x  + 0.5 * DT * vloc.x;
	midPosition.y = position.y  + 0.5 * DT * vloc.y;
	midPosition.z = position.z  + 0.5 * DT * vloc.z;
	float4 midVelocity = tex3D(texref_vel, midPosition.x*NX + 0.5, midPosition.y*NY + 0.5, midPosition.z*NZ + 0.5);

	newPosition.x = (position.x + DT * midVelocity.x);
	newPosition.y = (position.y + DT * midVelocity.y);
	newPosition.z = (position.z + DT * midVelocity.z);

	particle[index] = newPosition;
	__syncthreads();

}

__global__ void
advect_density_k(float *d, size_t pitch){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	if ((ex != 0) && (ex != (NX - 1)) && (ey != 0) && (ey != (NY - 1)) && (ez != 0) && (ez != (NZ - 1))){

		float4 velocity;
		float3 ploc;
		float den;

		//find the velocity of this position
		velocity = tex3D(texref_vel, ex + 0.5, ey + 0.5, ez + 0.5);

		//tracing back
		ploc.x = (ex) - velocity.x * DT * NX;
		ploc.y = (ey) - velocity.y * DT * NY;
		ploc.z = (ez) - velocity.z * DT * NZ;

		//get the density of tracing back position
		den = tex3D(texref_den, ploc.x + 0.5, ploc.y + 0.5, ploc.z + 0.5);
		d[ez*NX*NY + ey*NX + ex] = den;
	}

	__syncthreads();
}

extern "C"
void advectParticles(GLuint vbo, float4 *v, float *d)
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
	advectParticles_Runge_Kutta_k << <block_size, threads_size >> >(p, v, tPitch_v);
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


extern "C"
void advectDensity(float4 *v, float *d){
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	update_vel_texture(v, NX, NY, tPitch_v);
	update_1f_texture(array_den, d, NX, NY, tPitch_den);
	advect_density_k << <block_size, threads_size >> >(d, tPitch_den);
	bc_density_k << <block_size, threads_size >> >(d, tPitch_den, 1.f);
	getLastCudaError("advectDensity_k failed.");
}

///////////////////////////////////////////////////////////////////////////////////////////////
//welcome to the marker level set method<><><><><><><><><><><><><><><><><><><><><><><><><><><><
///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void
advect_levelset_k(float *ls, size_t pitch){
	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	float4 velocity;
	float3 ploc;
	float new_levelset;
	if (ex != 0 && ex != (LNX - 1) && ey != 0 && ey != (LNY - 1) && ez != 0 && ez != (LNZ - 1)){
		//	find the velocity of this position
		velocity = tex3D(texref_vel, ex * 0.5 + 0.5, ey * 0.5 + 0.5, ez * 0.5 + 0.5);

		//tracing back
		ploc.x = (ex)-DT * velocity.x * LNX;
		ploc.y = (ey)-DT * velocity.y * LNY;
		ploc.z = (ez)-DT * velocity.z * LNZ;

		//get the density of tracing back position
		new_levelset = tex3D(texref_levelset, ploc.x + 0.5, ploc.y + 0.5, ploc.z + 0.5);
		ls[ez*LNX*LNY + ey*LNX + ex] = new_levelset;

		__syncthreads();
	}
}

__global__ void
advect_levelset_BFECC_k(float *ls, size_t pitch){
	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	float4 velocity;
	float3 ploc;
	float new_levelset;

	float3 grid_position = make_float3(ex, ey, ez);

	if (ex != 0 && ex != (LNX - 1) && ey != 0 && ey != (LNY - 1) && ez != 0 && ez != (LNZ - 1)){
		//tracing backward
		velocity = tex3D(texref_vel, ex * 0.5 + 0.5, ey * 0.5 + 0.5, ez * 0.5 + 0.5);
		ploc.x = (ex) - DT * velocity.x * LNX;
		ploc.y = (ey) - DT * velocity.y * LNY;
		ploc.z = (ez) - DT * velocity.z * LNZ;
		
		//tracing forward
		velocity = tex3D(texref_vel, ploc.x * 0.5 + 0.5, ploc.y * 0.5 + 0.5, ploc.z * 0.5 + 0.5);
		ploc.x = ploc.x + DT * velocity.x * LNX;
		ploc.y = ploc.y + DT * velocity.y * LNY;
		ploc.z = ploc.z + DT * velocity.z * LNZ;

		grid_position = grid_position + 0.5*(grid_position - ploc);
		velocity = tex3D(texref_vel, grid_position.x * 0.5 + 0.5, grid_position.y * 0.5 + 0.5, grid_position.z * 0.5 + 0.5);
		ploc.x = grid_position .x - DT * velocity.x * LNX;
		ploc.y = grid_position .y - DT * velocity.y * LNY;
		ploc.z = grid_position .z - DT * velocity.z * LNZ;

		new_levelset = tex3D(texref_levelset, ploc.x + 0.5, ploc.y + 0.5, ploc.z + 0.5);
		ls[ez*LNX*LNY + ey*LNX + ex] = new_levelset;
		

		__syncthreads();
	}
}

extern "C"
void advectLevelSet(float4 *v, float *ls){
	dim3 block_size(LNX / THREAD_X, LNY / THREAD_Y, LNZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	update_vel_texture(v, NX, NY, tPitch_v);
	update_1f_texture(array_levelset, ls, LNX, LNY, tPitch_lsf);
	advect_levelset_k << <block_size, threads_size >> >(ls, tPitch_den);
//	bc_density_k << <block_size, threads_size >> >(ls, tPitch_den, 1.f);
	getLastCudaError("advectLevelSet_k failed.");
}

__global__ void
correctLevelset_first_k(float3 *p, float2 *con){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	float e_cons = 2.71828;
	float constant_c = 0.5;
	if (ez*NX*NY + ey*NX + ex > 31104) 
		return;//more than particle amount
	
	float3 particle_location = p[ez*NX*NY + ey*NX + ex];
	//convert to gird space
	particle_location = make_float3(particle_location.x * LNX, particle_location.y * LNY, particle_location.z * LNZ);

	float value_levelset = tex3D(texref_levelset, particle_location.x + 0.5, particle_location.y + 0.5, particle_location.z + 0.5);//get phi(k)
	if (value_levelset > 2 || value_levelset < -2)
		return;//too far away from zero level set

	int3 grid_location0;
	grid_location0.x = floor(particle_location.x);
	grid_location0.y = floor(particle_location.y);
	grid_location0.z = floor(particle_location.z);
	int3 grid_location[27];
	//bottom 9
	grid_location[0] = make_int3(grid_location0.x - 1, grid_location0.y - 1, grid_location0.z - 1);
	grid_location[1] = make_int3(grid_location0.x,	   grid_location0.y - 1, grid_location0.z - 1);
	grid_location[2] = make_int3(grid_location0.x + 1, grid_location0.y - 1, grid_location0.z - 1);
	grid_location[3] = make_int3(grid_location0.x - 1, grid_location0.y - 1, grid_location0.z);
	grid_location[4] = make_int3(grid_location0.x,	   grid_location0.y - 1, grid_location0.z);
	grid_location[5] = make_int3(grid_location0.x + 1, grid_location0.y - 1, grid_location0.z);
	grid_location[6] = make_int3(grid_location0.x - 1, grid_location0.y - 1, grid_location0.z + 1);
	grid_location[7] = make_int3(grid_location0.x,	   grid_location0.y - 1, grid_location0.z + 1);
	grid_location[8] = make_int3(grid_location0.x + 1, grid_location0.y - 1, grid_location0.z + 1);

	//middle 8
	grid_location[9] = make_int3(grid_location0.x - 1, grid_location0.y, grid_location0.z - 1);
	grid_location[10]= make_int3(grid_location0.x,	   grid_location0.y, grid_location0.z - 1);
	grid_location[11]= make_int3(grid_location0.x + 1, grid_location0.y, grid_location0.z - 1);
	grid_location[12]= make_int3(grid_location0.x - 1, grid_location0.y, grid_location0.z);
	grid_location[13]= make_int3(grid_location0.x + 1, grid_location0.y, grid_location0.z);
	grid_location[14]= make_int3(grid_location0.x - 1, grid_location0.y, grid_location0.z + 1);
	grid_location[15]= make_int3(grid_location0.x,	   grid_location0.y, grid_location0.z + 1);
	grid_location[16]= make_int3(grid_location0.x + 1, grid_location0.y, grid_location0.z + 1);

	//top 9
	grid_location[17] = make_int3(grid_location0.x - 1, grid_location0.y + 1, grid_location0.z - 1);
	grid_location[18] = make_int3(grid_location0.x,	    grid_location0.y + 1, grid_location0.z - 1);
	grid_location[19] = make_int3(grid_location0.x + 1, grid_location0.y + 1, grid_location0.z - 1);
	grid_location[20] = make_int3(grid_location0.x - 1, grid_location0.y + 1, grid_location0.z);
	grid_location[21] = make_int3(grid_location0.x,	    grid_location0.y + 1, grid_location0.z);
	grid_location[22] = make_int3(grid_location0.x + 1, grid_location0.y + 1, grid_location0.z);
	grid_location[23] = make_int3(grid_location0.x - 1, grid_location0.y + 1, grid_location0.z + 1);
	grid_location[24] = make_int3(grid_location0.x,	    grid_location0.y + 1, grid_location0.z + 1);
	grid_location[25] = make_int3(grid_location0.x + 1, grid_location0.y + 1, grid_location0.z + 1);
	//center
	grid_location[26] = grid_location0;
	
	for (int i = 0; i < 27; i++){
		if (grid_location[i].x >= 0 && grid_location[i].x < LNX &&
			grid_location[i].y >= 0 && grid_location[i].y < LNY &&
			grid_location[i].z >= 0 && grid_location[i].z < LNZ){
			float sq_dis = 
				(particle_location.x - grid_location[i].x)*(particle_location.x - grid_location[i].x) +
				(particle_location.y - grid_location[i].y)*(particle_location.y - grid_location[i].y) +
				(particle_location.z - grid_location[i].z)*(particle_location.z - grid_location[i].z);
			if (sq_dis < 1){
				//calculate contribution of this particle
				float q_c = -sq_dis / constant_c / constant_c;
				float _c = -1.f / constant_c / constant_c;
				float w = (pow(e_cons, q_c) - pow(e_cons, _c)) / (1 - pow(e_cons, _c));//get w(k)
				//float value_levelset = tex3D(texref_levelset, 
				//	particle_location.x + 0.5, particle_location.y + 0.5, particle_location.z + 0.5);//get phi(k)
				con[grid_location[i].z*LNX*LNY + grid_location[i].y*LNX + grid_location[i].x].x += value_levelset * w;
				con[grid_location[i].z*LNX*LNY + grid_location[i].y*LNX + grid_location[i].x].y += w;
			}
		}
	}
	__syncthreads();
}

__global__ void
correctLevelset_second_k(float *ls, float2 *con){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;
	//phi(new) = phi(i) - con(i) / w(i);	
	if (con[ez*LNX*LNY + ey*LNX + ex].x != 0 && con[ez*LNX*LNY + ey*LNX + ex].y != 0){
		//influence level set function where have particles
		float newdata = ls[ez*LNX*LNY + ey*LNX + ex] - 
			con[ez*LNX*LNY + ey*LNX + ex].x / con[ez*LNX*LNY + ey*LNX + ex].y;
		ls[ez*LNX*LNY + ey*LNX + ex] = newdata;
	}
}
__device__ float
minmod(float a, float b){
	return abs(a) < abs(b) ? a : b;
}
#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif



#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

__global__ void
reinit_Levelset_k(float *ls){

	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	/*if (ex == 0 || ex == (LNX - 1) || ey == 0 || ey == (LNY - 1) || ez == 0 || ez == (LNZ - 1))
		return;*/
	float grid_value = ls[ez*LNX*LNY + ey*LNX + ex];

	/*float left_value = ls[ez*LNX*LNY + ey*LNX + ex - 1];
	float righ_value = ls[ez*LNX*LNY + ey*LNX + ex + 1];
	float bott_value = ls[ez*LNX*LNY + (ey-1)*LNX + ex];
	float topp_value = ls[ez*LNX*LNY + (ey+1)*LNX + ex];
	float back_value = ls[(ez-1)*LNX*NY + ey*LNX + ex];
	float fron_value = ls[(ez+1)*LNX*NY + ey*LNX + ex];*/
	//if (abs(left_value - grid_value) < 1.1 && abs(righ_value - grid_value) < 1.1 &&
	//	abs(topp_value - grid_value) < 1.1 && abs(bott_value - grid_value) < 1.1 &&
	//	abs(fron_value - grid_value) < 1.1 && abs(back_value - grid_value) < 1.1 &&
	//	abs(left_value - grid_value) > 0.9 && abs(righ_value - grid_value) > 0.9 &&
	//	abs(topp_value - grid_value) > 0.9 && abs(bott_value - grid_value) > 0.9 &&
	//	abs(fron_value - grid_value) > 0.9 && abs(back_value - grid_value) > 0.9){
	//	//slope is not sufficiently high
	//	return;
	//}

	int3 grid_location0 = make_int3(ex, ey, ez);
	int3 grid_location[26];
	//bottom 9
	grid_location[0] = make_int3(grid_location0.x - 1, grid_location0.y - 1, grid_location0.z - 1);
	grid_location[1] = make_int3(grid_location0.x, grid_location0.y - 1, grid_location0.z - 1);
	grid_location[2] = make_int3(grid_location0.x + 1, grid_location0.y - 1, grid_location0.z - 1);
	grid_location[3] = make_int3(grid_location0.x - 1, grid_location0.y - 1, grid_location0.z);
	grid_location[4] = make_int3(grid_location0.x, grid_location0.y - 1, grid_location0.z);
	grid_location[5] = make_int3(grid_location0.x + 1, grid_location0.y - 1, grid_location0.z);
	grid_location[6] = make_int3(grid_location0.x - 1, grid_location0.y - 1, grid_location0.z + 1);
	grid_location[7] = make_int3(grid_location0.x, grid_location0.y - 1, grid_location0.z + 1);
	grid_location[8] = make_int3(grid_location0.x + 1, grid_location0.y - 1, grid_location0.z + 1);

	//middle 8
	grid_location[9] = make_int3(grid_location0.x - 1, grid_location0.y, grid_location0.z - 1);
	grid_location[10] = make_int3(grid_location0.x, grid_location0.y, grid_location0.z - 1);
	grid_location[11] = make_int3(grid_location0.x + 1, grid_location0.y, grid_location0.z - 1);
	grid_location[12] = make_int3(grid_location0.x - 1, grid_location0.y, grid_location0.z);
	grid_location[13] = make_int3(grid_location0.x + 1, grid_location0.y, grid_location0.z);
	grid_location[14] = make_int3(grid_location0.x - 1, grid_location0.y, grid_location0.z + 1);
	grid_location[15] = make_int3(grid_location0.x, grid_location0.y, grid_location0.z + 1);
	grid_location[16] = make_int3(grid_location0.x + 1, grid_location0.y, grid_location0.z + 1);

	//top 9
	grid_location[17] = make_int3(grid_location0.x - 1, grid_location0.y + 1, grid_location0.z - 1);
	grid_location[18] = make_int3(grid_location0.x, grid_location0.y + 1, grid_location0.z - 1);
	grid_location[19] = make_int3(grid_location0.x + 1, grid_location0.y + 1, grid_location0.z - 1);
	grid_location[20] = make_int3(grid_location0.x - 1, grid_location0.y + 1, grid_location0.z);
	grid_location[21] = make_int3(grid_location0.x, grid_location0.y + 1, grid_location0.z);
	grid_location[22] = make_int3(grid_location0.x + 1, grid_location0.y + 1, grid_location0.z);
	grid_location[23] = make_int3(grid_location0.x - 1, grid_location0.y + 1, grid_location0.z + 1);
	grid_location[24] = make_int3(grid_location0.x, grid_location0.y + 1, grid_location0.z + 1);
	grid_location[25] = make_int3(grid_location0.x + 1, grid_location0.y + 1, grid_location0.z + 1);
	for (int i = 0; i < 26; i++){
		if (grid_location[i].x >= 0 && grid_location[i].x < LNX &&
			grid_location[i].y >= 0 && grid_location[i].y < LNY &&
			grid_location[i].z >= 0 && grid_location[i].z < LNZ){
			float value = ls[grid_location[i].z*LNX*LNY + grid_location[i].y*LNX + grid_location[i].x];
			if (value*grid_value < 0 && abs(grid_value - value) < 1)
				//grid point is close to the interface
				return;
		}
	}

	//deal with reinitialize

	float righ,left,uppp,down,fron,back;
	if (ex < LNX - 1)
		righ = ls[ez*LNX*LNY + ey*LNX + (ex + 1)];
	else
		righ = grid_value;
	if (ex > 0)
		left = ls[ez*LNX*LNY + ey*LNX + (ex - 1)];
	else
		left = grid_value;
	if (ey < LNY - 1)
		uppp = ls[ez*LNX*LNY + (ey + 1)*LNX + ex];
	else
		uppp = grid_value;
	if (ey > 0)
		down = ls[ez*LNX*LNY + (ey - 1)*LNX + ex];
	else
		down = grid_value;
	if (ez < LNZ - 1)
		fron = ls[(ez + 1)*LNX*LNY + ey*LNX + ex];
	else
		fron = grid_value;
	if (ez > 0)
		back = ls[(ez - 1)*LNX*LNY + ey*LNX + ex];
	else
		back = grid_value;
	/*float left = ls[ez*LNX*LNY + ey*LNX + (ex - 1)];
	float uppp = ls[ez*LNX*LNY + (ey + 1)*LNX + ex];
	float down = ls[ez*LNX*LNY + (ey - 1)*LNX + ex];
	float fron = ls[(ez + 1)*LNX*LNY + ey*LNX + ex];
	float back = ls[(ez - 1)*LNX*LNY + ey*LNX + ex];*/

	float dx1 = righ - grid_value;
	float dx2 = grid_value - left;
	float dy1 = uppp - grid_value;
	float dy2 = grid_value - down;
	float dz1 = fron - grid_value;
	float dz2 = grid_value - back;

	float si = grid_value* rsqrt(grid_value*grid_value + 1);

	float a1 = max(dx1, 0);
	float a2 = min(dx1, 0);
	float b1 = max(dx2, 0);
	float b2 = min(dx2, 0);
	float c1 = max(dy1, 0);
	float c2 = min(dy1, 0);
	float d1 = max(dy2, 0);
	float d2 = min(dy2, 0);
	float e1 = max(dz1, 0);
	float e2 = min(dz1, 0);
	float f1 = max(dz2, 0);
	float f2 = min(dz2, 0);
	
	float hg;
	if (si < 0)
		hg = sqrt(max(a1*a1, b2*b2) + max(c1*c1, d2*d2) + max(e1*e1 , f2*f2)) - 1;
	else															
		hg = sqrt(max(a2*a2, b1*b1) + max(c2*c2, d1*d1) + max(e2*e2 , f1*f1)) - 1;

	ls[ez*LNX*LNY + ey*LNX + ex] = grid_value - DT*si*hg;
	////////////////////////////////////////////////////////////////////////////////////////
//	int si = 0;
////	si = grid_value > 0 ? 1 : -1;
//	si = grid_value * rsqrt(grid_value*grid_value + 1);
////	float3 dis = make_float3(si *0.57735, si *0.57735, si *0.57735);
//	float3 dis = make_float3(si, si, si);
//	float3 cur_loc = make_float3(ex, ey, ez);
//	float3 pre_loc = cur_loc - dis;
//	float value_levelset = tex3D(texref_levelset,
//		pre_loc.x + 0.5, pre_loc.y + 0.5, pre_loc.z + 0.5);//get phi(pre)
//	ls[ez*LNX*LNY + ey*LNX + ex] = value_levelset + si;
}

extern "C"
void correctLevelSet(float *ls, float2 *con){
	dim3 block_size(LNX / THREAD_X, LNY / THREAD_Y, LNZ / THREAD_Z);

	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);
	//get location of particles>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	float3 *particle;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&particle, &num_bytes, cuda_vbo_resource);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");
	
	cudaMemset(con, 0, sizeof(float2)*LNX*LNY*LNZ);//reset contribution data
	update_1f_texture(array_levelset, ls, LNX, LNY, tPitch_lsf);
	correctLevelset_first_k << <block_size, threads_size >> >(particle, con);
	correctLevelset_second_k << <block_size, threads_size >> >(ls, con);
	for (int i = 0; i < 5; i++){
		update_1f_texture(array_levelset, ls, LNX, LNY, tPitch_lsf);
		reinit_Levelset_k << <block_size, threads_size >> >(ls);
	}
//	bc_levelset_k << <block_size, threads_size >> >(ls, tPitch_lsf, 1.f);
	getLastCudaError("advectParticles_k failed.");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");
}


///////////////////////////////////////////////////////////////////////////////////////////////
//welcome to ray casting method<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void
raycasting_k(int maxx, int maxy, float *ls, float4 *intersection, float3 *normal, float3 camera){
	int ex = threadIdx.x + blockIdx.x * blockDim.x;
	int ey = threadIdx.y + blockIdx.y * blockDim.y;
	if (ex > maxx || ey > maxy) return;//cuda out of range

	float4 pos4 = tex2D(texref_ray, ex + 0.5, ey + 0.5);
	if (pos4.w < 0.5) return;//there is no ray there
	
	float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
	float3 dir = pos - camera;//the direction from camera to ray inter in model space
	dir = normalize(dir);
	float advect_value0 = 0;//previous
	float advect_value;
	int reverse = 0;
	int counter = 0;
	while (1){
		if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
			pos.x > 1 || pos.y > 1 || pos.z > 1){
			//ray out of volume
			return;
		}
		//get the levelset value at pos
		advect_value = tex3D(texref_levelset, pos.x*LNX + 0.5, pos.y*LNY + 0.5, pos.z*LNZ + 0.5);
		float d = 0.05;
		if (advect_value < 0.01 && counter == 0){
			//zero level is on the boundary of entire volume
			intersection[ey * 1024 + ex] = make_float4(pos.x, pos.y, pos.z, 1);
			if (pos.x < 0.05)
				normal[ey * 1024 + ex] = make_float3(-1, 0, 0);
			if (pos.x > 0.95)
				normal[ey * 1024 + ex] = make_float3(1, 0, 0);
			if (pos.y < 0.05)
				normal[ey * 1024 + ex] = make_float3(0, -1, 0);
			if (pos.y > 0.95)
				normal[ey * 1024 + ex] = make_float3(0, 1, 0);
			if (pos.z < 0.05)
				normal[ey * 1024 + ex] = make_float3(0, 0, -1);
			if (pos.z > 0.95)
				normal[ey * 1024 + ex] = make_float3(0, 0, 1);
			return;
		}
		
		if (counter == 0 && advect_value < 0){
			//first pos in water
			intersection[ey * 1024 + ex] = make_float4(pos.x, pos.y, pos.z, 1);
			float left = tex3D(texref_levelset, pos.x*LNX + 0.5-d, pos.y*LNY + 0.5  , pos.z*LNZ + 0.5);
			float righ = tex3D(texref_levelset, pos.x*LNX + 0.5+d, pos.y*LNY + 0.5  , pos.z*LNZ + 0.5);
			float bott = tex3D(texref_levelset, pos.x*LNX + 0.5  , pos.y*LNY + 0.5-d, pos.z*LNZ + 0.5);
			float topp = tex3D(texref_levelset, pos.x*LNX + 0.5  , pos.y*LNY + 0.5+d, pos.z*LNZ + 0.5);
			float back = tex3D(texref_levelset, pos.x*LNX + 0.5  , pos.y*LNY + 0.5  , pos.z*LNZ + 0.5-d);
			float fron = tex3D(texref_levelset, pos.x*LNX + 0.5  , pos.y*LNY + 0.5  , pos.z*LNZ + 0.5+d);
			float3 g = make_float3(righ - left, topp - bott, fron - back);
			g = normalize(g);
			normal[ey * 1024 + ex] = g;
			return;
		}
		if (advect_value < 0.01){
			//the value is less than threshold
			intersection[ey * 1024 + ex] = make_float4(pos.x, pos.y, pos.z, 1);
			float left = tex3D(texref_levelset, pos.x*LNX + 0.5 - d, pos.y*LNY + 0.5, pos.z*LNZ + 0.5);
			float righ = tex3D(texref_levelset, pos.x*LNX + 0.5 + d, pos.y*LNY + 0.5, pos.z*LNZ + 0.5);
			float bott = tex3D(texref_levelset, pos.x*LNX + 0.5, pos.y*LNY + 0.5 - d, pos.z*LNZ + 0.5);
			float topp = tex3D(texref_levelset, pos.x*LNX + 0.5, pos.y*LNY + 0.5 + d, pos.z*LNZ + 0.5);
			float back = tex3D(texref_levelset, pos.x*LNX + 0.5, pos.y*LNY + 0.5, pos.z*LNZ + 0.5 - d);
			float fron = tex3D(texref_levelset, pos.x*LNX + 0.5, pos.y*LNY + 0.5, pos.z*LNZ + 0.5 + d);
			float3 g = make_float3(righ - left, topp - bott, fron - back);
			g = normalize(g);
			normal[ey * 1024 + ex] = g;
			return;
		}
		if (counter != 0 && advect_value*advect_value0 < 0){
			// from air to water, go over the zero level set
			float t = advect_value / (advect_value - advect_value0);
			pos = pos - t * advect_value * dir / LNX;
			intersection[ey * 1024 + ex] = make_float4(pos.x, pos.y, pos.z, 1);
			float left = tex3D(texref_levelset, pos.x*LNX + 0.5 - d, pos.y*LNY + 0.5, pos.z*LNZ + 0.5);
			float righ = tex3D(texref_levelset, pos.x*LNX + 0.5 + d, pos.y*LNY + 0.5, pos.z*LNZ + 0.5);
			float bott = tex3D(texref_levelset, pos.x*LNX + 0.5, pos.y*LNY + 0.5 - d, pos.z*LNZ + 0.5);
			float topp = tex3D(texref_levelset, pos.x*LNX + 0.5, pos.y*LNY + 0.5 + d, pos.z*LNZ + 0.5);
			float back = tex3D(texref_levelset, pos.x*LNX + 0.5, pos.y*LNY + 0.5, pos.z*LNZ + 0.5 - d);
			float fron = tex3D(texref_levelset, pos.x*LNX + 0.5, pos.y*LNY + 0.5, pos.z*LNZ + 0.5 + d);
			float3 g = make_float3(righ - left, topp - bott, fron - back);
			g = normalize(g);
			normal[ey * 1024 + ex] = g;
			return;
		}
		pos = pos + advect_value * dir / LNX;
		advect_value0 = advect_value;
		counter++;
	}
	__syncthreads();
}

extern "C"
void raycasting(int x, int y, float *ls, float3 camera){
	dim3 block_size(128, 128);
	dim3 threads_size(8, 8);

	float4 *intersection;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_intersection, 0));
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&intersection, &num_bytes, cuda_vbo_intersection));
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	float3 *normal;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_normal, 0));
	getLastCudaError("cudaGraphicsMapResources failed");

	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&normal, &num_bytes, cuda_vbo_normal));
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	update_1f_texture(array_levelset, ls, LNX, LNY, tPitch_lsf);
	checkCudaErrors(cudaMemset(intersection, 0, sizeof(float4) * 1024 * 1024));//reset intersection data
	checkCudaErrors(cudaMemset(normal, 0, sizeof(float3) * 1024 * 1024));//reset intersection data
	getLastCudaError("cudaGraphicsUnmapResources failed");
	raycasting_k << <block_size, threads_size >> >(x, y, ls, intersection, normal, camera);
	/*cudaMemcpy(hintersection, intersection, sizeof(float4) * 1024 * 1024, cudaMemcpyDeviceToHost);
	for (int i = 500; i < 600; i++){
		printf("(%f,%f,%f,%f)\n", hintersection[1024 * 512 + i].x, hintersection[1024 * 512 + i].y, hintersection[1024 * 512 + i].z, hintersection[1024 * 512 + i].w);
	}
	printf("====================================================\n");*/
	/*cudaMemcpy(hnormal, normal, sizeof(float3) * 1024 * 1024, cudaMemcpyDeviceToHost);
	for (int j = 0; j < 1024; j++){
		for (int i = 0; i < 1024; i++){
			if (hnormal[1024 * j + i].y!=0)
				if (i>500 && i<520 && j>500 && j<520)
					printf("(%f,%f,%f)\n", hnormal[1024 * j + i].x, hnormal[1024 * j + i].y, hnormal[1024 * j + i].z);
		}
	}
	printf("====================================================\n");*/
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_intersection, 0));
	getLastCudaError("cudaGraphicsUnmapResources failed");
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_normal, 0));
	getLastCudaError("cudaGraphicsUnmapResources failed");
}




__global__ void
add_source_k(float4 *v, float *d, float *l, int x, int y, int z, int size){
	int ex = threadIdx.x + blockIdx.x * 8;
	int ey = threadIdx.y + blockIdx.y * 8;
	int ez = threadIdx.z + blockIdx.z * 8;

	int far_x = x + size;
	int far_y = y + size;
	int far_z = z + size;
	if (x >= 0 && far_x < NX && y >= 0 && far_y < NY && z >= 0 && far_z < NZ){
		//location is inside the volume
		if (ex >= x && ex <= far_x && ey >= y && ey <= far_y && ez >= z && ez <= far_z){
			//this thread is inside the location
			v[ez*NX*NY + ey*NX + ex].x -= 0.4;
		//	d[ez*NX*NY + ey*NX + ex] += 10.f;
		//	l[ez*NX*NY + ey*NX + ex] = -1;
		}
	}
}

extern "C"
void addSource(float4 *v, float *d, float*l, int dx, int dy, int dz, float size){
	dim3 block_size(NX / THREAD_X, NY / THREAD_Y, NZ / THREAD_Z);
	dim3 threads_size(THREAD_X, THREAD_Y, THREAD_Z);

	add_source_k << <block_size, threads_size >> >(v, d, l, dx, dy, dz, size);
	reinit_Levelset_k << <block_size, threads_size >> >(l);
	getLastCudaError("add_source_k failed.");
}