#include "Water.h"
using namespace Rendering;
using namespace Models;




float4 *hvfield = NULL;
extern float4 *dvfield = NULL;
float4 *dtemp = NULL;
float4 *dpressure = NULL;
float4 *ddivergence = NULL;

GLuint vbo = 0;                 // OpenGL vertex buffer object
struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange

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
//	memset(hvfield, 0, sizeof(float4) * DS);

	// Allocate and initialize device data
	cudaMallocPitch((void **)&dvfield, &tPitch_v, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&dtemp, &tPitch_t, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&ddivergence, &tPitch_d, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&dpressure, &tPitch_p, sizeof(float4)*NX*NY, NZ);

//	cudaMemcpy(dvfield, hvfield, sizeof(float4) * DS, cudaMemcpyHostToDevice);
//	cudaMemcpy(dtemp, hvfield, sizeof(float4)* DS, cudaMemcpyHostToDevice);
	

	initParticles_velocity(hvfield, dvfield);
	setupTexture(NX,NY,NZ);
	bindTexture();

	// Create particle array
	particles = (float3 *)malloc(sizeof(float3) * DS);
	memset(particles, 0, sizeof(float3) * DS);

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
	glBindVertexArray(0);
	this->vao = vao;
	this->vbos.push_back(vbo);
	texture = SOIL_load_OGL_texture("../Textures/water_particle.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
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
//	addForces(dvfield, DIM, DIM, 10, 10, FORCE * DT * 0.01, FORCE * DT * 0.01, FR);
	glUseProgram(program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	//*Matrix4::Scale(Vector3(10, 10, 10))
	Matrix4 modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10));
	//	modelMatrix.SetScalingVector(Vector3(10, 10, 10));
	//	std::cout << viewMatrix.GetPositionVector() << std::endl;
	glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "viewMatrix"), 1, false, (float*)&viewMatrix);

	glUniform1i(glGetUniformLocation(program, "diffuse_texture"), 0);
	glPointSize(6);
	glBindVertexArray(vao);

//	glEnable(GL_POINT_SMOOTH);
//	glEnable(GL_BLEND);
//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
////	glEnableClientState(GL_VERTEX_ARRAY);
//	glDisable(GL_DEPTH_TEST);
//	glDisable(GL_CULL_FACE);

	glDrawArrays(GL_POINTS, 0, DS);
	glUseProgram(0);
	/*glColor4f(0, 1, 0, 1);
	glPointSize(1);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glBindBuffer(GL_ARRAY_BUFFER_ARB, vbo);
	glVertexPointer(2, GL_FLOAT, 0, NULL);
	glDrawArrays(GL_POINTS, 0, DS);
	glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisable(GL_TEXTURE_2D);*/
}


#define MYRAND (rand() / (float)RAND_MAX)

void Water::initParticles_velocity(float4 *h, float4 *d){
	int i, j, k;
	for (k = 0; k < NZ; k++){

		for (i = 0; i < NY; i++)
		{
			for (j = 0; j < NX; j++)
			{
				if (j>30 && j<40 && i>30 && i<40 &&4 == k){
					h[k*NX*NY + i*NX + j].x = 0.5;
					h[k*NX*NY + i*NX + j].y = 1;
					h[k*NX*NY + i*NX + j].z = 0;
				}
				else{
					h[k*NX*NY + i*NX + j].x = 0;
					h[k*NX*NY + i*NX + j].y = 0;
					h[k*NX*NY + i*NX + j].z = 0;
				}
			}
		}
	}
	
	cudaMemcpy(d, h, sizeof(float4)* DS, cudaMemcpyHostToDevice);
}


void Water::initParticles(float3 *p, int dx, int dy, int dz){
	int i, j, k;
	for (k = 0; k < dz; k++){

		for (i = 0; i < dy; i++)
		{
			for (j = 0; j < dx; j++)
			{
				p[k*dx*dy + i*dx + j].x = (float)(0.8*j + MYRAND) / dx;
				p[k*dx*dy + i*dx + j].y = (float)(0.8*i + MYRAND) / dy;
				p[k*dx*dy + i*dx + j].z = (float)(0.8*k + MYRAND) / dz;
			}
		}
	}
}

void Water::simulateFluids(void)
{
	// simulate fluid
	advect(dvfield, dtemp, NX, NY, NZ, DT);
	diffuse(dvfield, dtemp, NX, NY, NZ, DT);
	projection(dvfield, dtemp, dpressure, ddivergence, NX, NY, NZ, DT);
	advectParticles(vbo, dvfield, NX, NY, NZ, DT);
}


