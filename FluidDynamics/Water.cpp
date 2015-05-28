#include "Water.h"
using namespace Rendering;
using namespace Models;




float4 *hvfield = NULL;
extern float4 *dvfield = NULL;
float4 *dtemp = NULL;
float4 *dpressure = NULL;
float4 *ddivergence = NULL;
GLuint vbo2 = 0;                 // OpenGL vertex buffer object
GLuint vbo1 = 0;                 // OpenGL vertex buffer object
GLuint vbo = 0;                 // OpenGL vertex buffer object
struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
struct cudaGraphicsResource *cuda_vbo_resource1; // handles OpenGL-CUDA exchange

// Texture pitch
size_t tPitch_v = 0; // Now this is compatible with gcc in 64-bit

// Texture pitch
size_t tPitch_t = 0; // Now this is compatible with gcc in 64-bit
// Texture pitch
size_t tPitch_d = 0; // Now this is compatible with gcc in 64-bit
// Texture pitch
size_t tPitch_p = 0; // Now this is compatible with gcc in 64-bit

extern "C"
void advect(float4 *v, float4 *temp, int dx, int dy, int dz, float dt);

extern "C"
void diffuse(float4 *v, float4 *temp, int dx, int dy, int dz, float dt);

extern "C"
void projection(float4 *v, float4 *temp, float4 *pressure, float4* divergence, int dx, int dy, int dz, float dt);

extern "C"
void advectParticles(GLuint vbo, float4 *v, int dx, int dy, int dz, float dt);

Water::Water()
{
	position = Vector3(0, 0, 0);
//	orientation = Quaternion::AxisAngleToQuaterion(Vector3(1,0,0),180);
}

Water::~Water()
{
	//is going to be deleted in Models.cpp (inheritance)
	cudaGraphicsUnregisterResource(cuda_vbo_resource);

	unbindTexture();
//	deleteTexture();

	// Free all host and device resources
	free(hvfield);
	
	free(particles);
	cudaFree(dvfield);
	cudaFree(dtemp);
	cudaFree(dpressure);
	cudaFree(ddivergence);

}

void Water::Create()
{

	GLint bsize;
	tPitch_v = 0;
	tPitch_t = 0;
	tPitch_p = 0;
	tPitch_d = 0;

	int devID;
	cudaDeviceProp deviceProps;
	int fakeargc = 1;
	char *fakeargv[] = { "fake", NULL };
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	devID = findCudaGLDevice(fakeargc, (const char **)fakeargv);

	// get number of SMs on this GPU
	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
	printf("CUDA device [%s] has %d Multi-Processors\n",
		deviceProps.name, deviceProps.multiProcessorCount);

	hvfield = (float4 *)malloc(sizeof(float4) * DS);


	// Allocate and initialize device data
	cudaMallocPitch((void **)&dvfield, &tPitch_v, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&dtemp, &tPitch_t, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&ddivergence, &tPitch_d, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&dpressure, &tPitch_p, sizeof(float4)*NX*NY, NZ);
//	memset(hvfield, 0, sizeof(float4) * DS);
//	cudaMemcpy(dvfield, hvfield, sizeof(float4) * DS, cudaMemcpyHostToDevice);
//	cudaMemcpy(dtemp, hvfield, sizeof(float4)* DS, cudaMemcpyHostToDevice);
	

	initParticles_velocity(hvfield, dvfield);
	setupTexture(NX,NY,NZ);
	bindTexture();

	

	

	//for paritcles system>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	// Create particle array
	particles = (float3 *)malloc(sizeof(float3) * DS);
	memset(particles, 0.5, sizeof(float3) * DS);
	initParticles(particles, NX, NY, NZ);

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * DS, particles, GL_DYNAMIC_DRAW);

	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

	if (bsize != (sizeof(float3) * DS))
		return;

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	this->vao = vao;
	this->vbos.push_back(vbo);

	
	//for velocity field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	v_position = (float3 *)malloc(sizeof(float3) * DS);
	initVelocityPosition(v_position, NX, NY, NZ);

	glGenVertexArrays(1, &vao1);
	glBindVertexArray(vao1);


	//velocity position
	glGenBuffers(1, &vbo1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo1);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * DS, v_position, GL_DYNAMIC_DRAW);
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);
	if (bsize != (sizeof(float3) * DS))
		return;

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
	this->vbos.push_back(vbo1);

	//velocity value
	glGenBuffers(1, &vbo2);
	glBindBuffer(GL_ARRAY_BUFFER, vbo2);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * DS, hvfield, GL_DYNAMIC_DRAW);
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);
	if (bsize != (sizeof(float4) * DS))
		return;

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);

	

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	this->vbos.push_back(vbo2);

	this->vao1 = vao1;
	

	texture = SOIL_load_OGL_texture("../Textures/water_particle.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);


	//bind particle position vbo
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
	//bind velocity value vbo
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource1, vbo2, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
	int CUDAVersion ;
	cudaRuntimeGetVersion(&CUDAVersion);
	std::cout << "CUDA version: "<< CUDAVersion << std::endl;
}


void Water::Update(Matrix4 viewMatrix)
{
	
	simulateFluids();
	
	Model::Update(viewMatrix);
}

void Water::Draw()
{
	//draw particle field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	glUseProgram(program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	//*Matrix4::Scale(Vector3(10, 10, 10))
	Matrix4 modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10));

	glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "viewMatrix"), 1, false, (float*)&viewMatrix);

	glUniform1i(glGetUniformLocation(program, "diffuse_texture"), 0);
	glPointSize(6);
	glBindVertexArray(vao);

	glDrawArrays(GL_POINTS, 0, DS);
	glUseProgram(0);

	//draw velocity field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	glUseProgram(program1);

	
	//*Matrix4::Scale(Vector3(10, 10, 10))
	modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(-1, 0, 0));

	glUniformMatrix4fv(glGetUniformLocation(program1, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program1, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program1, "viewMatrix"), 1, false, (float*)&viewMatrix);

	glPointSize(1);
	glBindVertexArray(vao1);



	glDrawArrays(GL_POINTS, 0, DS);
	glUseProgram(0);


}


#define MYRAND (rand() / (float)RAND_MAX)

void Water::initParticles_velocity(float4 *h, float4 *d){
	int i, j, k;
	for (k = 0; k < NZ; k++){

		for (i = 0; i < NY; i++)
		{
			for (j = 0; j < NX; j++)
			{
				if (j>3 && j<30 && i>15 && i<20 && k>15 && k<20){
					h[k*NX*NY + i*NX + j].x = 1;
					h[k*NX*NY + i*NX + j].y = 1;
					h[k*NX*NY + i*NX + j].z = 0;
				}
				else{
					h[k*NX*NY + i*NX + j].x = 1;
					h[k*NX*NY + i*NX + j].y = 1;
					h[k*NX*NY + i*NX + j].z = 0;
				}
			}
		}
	}
	
	cudaMemcpy(d, h, sizeof(float4)* DS, cudaMemcpyHostToDevice);
}


void Water::initParticles(float3 *p, int dx, int dy, int dz){
	int i, j, k;
	for (k = 1; k < (dz-1); k++){

		for (i = 1; i < (dy-1); i++)
		{
			for (j = 1; j < (dx-1); j++)
			{
				p[k*dx*dy + i*dx + j].x = (float)(j + 0.5) / dx;
				p[k*dx*dy + i*dx + j].y = (float)(i + 0.5) / dy;
				p[k*dx*dy + i*dx + j].z = (float)(k + 0.5) / dz;
			}
		}
	}
}
void Water::initVelocityPosition(float3 *vp, int dx, int dy, int dz){
	int i, j, k;
	for (k = 1; k < (dz - 1); k++){

		for (i = 1; i < (dy - 1); i++)
		{
			for (j = 1; j < (dx - 1); j++)
			{
				vp[k*dx*dy + i*dx + j].x = (float)(j + 0.5) / dx;
				vp[k*dx*dy + i*dx + j].y = (float)(i + 0.5) / dy;
				vp[k*dx*dy + i*dx + j].z = (float)(k + 0.5) / dz;
			}
		}
	}
}


void Water::simulateFluids(void)
{
	// simulate fluid
	advect(dvfield, dtemp, NX, NY, NZ, DT);
//	diffuse(dvfield, dtemp, NX, NY, NZ, DT);
//	projection(dvfield, dtemp, dpressure, ddivergence, NX, NY, NZ, DT);
	advectParticles(vbo, dvfield, NX, NY, NZ, DT);
}


