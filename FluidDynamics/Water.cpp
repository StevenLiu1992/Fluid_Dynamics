#include "Water.h"
using namespace Rendering;
using namespace Models;


cufftHandle planr2c;
cufftHandle planc2r;
static float2 *vxfield = NULL;
static float2 *vyfield = NULL;

float2 *hvfield = NULL;
float2 *dvfield = NULL;

GLuint vbo = 0;                 // OpenGL vertex buffer object
struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit

extern "C" void addForces(float2 *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);
extern "C" void advectVelocity(float2 *v, float *vx, float *vy, int dx, int pdx, int dy, float dt);
extern "C" void diffuseProject(float2 *vx, float2 *vy, int dx, int dy, float dt, float visc);
extern "C" void updateVelocity(float2 *v, float *vx, float *vy, int dx, int pdx, int dy);
extern "C" void advectParticles(GLuint vbo, float2 *v, int dx, int dy, float dt);

Water::Water()
{
	position = Vector3(0, 0, 0);
}

Water::~Water()
{
	//is going to be deleted in Models.cpp (inheritance)
	cudaGraphicsUnregisterResource(cuda_vbo_resource);

	unbindTexture();
	deleteTexture();

	// Free all host and device resources
	free(hvfield);
	free(particles);
	cudaFree(dvfield);
	cudaFree(vxfield);
	cudaFree(vyfield);
	cufftDestroy(planr2c);
	cufftDestroy(planc2r);
}

void Water::Create()
{

	GLint bsize;
	tPitch = 0;

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

	hvfield = (float2 *)malloc(sizeof(float2) * DS);
	memset(hvfield, 0, sizeof(float2) * DS);

	// Allocate and initialize device data
	cudaMallocPitch((void **)&dvfield, &tPitch, sizeof(float2)*DIM, DIM);

	cudaMemcpy(dvfield, hvfield, sizeof(float2) * DS,
		cudaMemcpyHostToDevice);
	// Temporary complex velocity field data
	cudaMalloc((void **)&vxfield, sizeof(float2) * PDS);
	cudaMalloc((void **)&vyfield, sizeof(float2) * PDS);


	setupTexture(DIM, DIM);
	bindTexture();

	// Create particle array
	particles = (float2 *)malloc(sizeof(float2) * DS);
	memset(particles, 0, sizeof(float2) * DS);

	initParticles(particles, DIM, DIM);

	// Create CUFFT transform plan configuration
	checkCudaErrors(cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R));
	// TODO: update kernels to use the new unpadded memory layout for perf
	// rather than the old FFTW-compatible layout
	cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_FFTW_PADDING);
	cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_FFTW_PADDING);
	
	
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * DS, particles, GL_DYNAMIC_DRAW);

	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

	if (bsize != (sizeof(float2) * DS))
		return;

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	this->vao = vao;
	this->vbos.push_back(vbo);
	texture = SOIL_load_OGL_texture("../Textures/water_particle.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

}


void Water::Update(Matrix4 viewMatrix)
{
	
	simulateFluids();
	
	Model::Update(viewMatrix);
}

void Water::Draw()
{
	glUseProgram(program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	Matrix4 modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10));
	//	modelMatrix.SetScalingVector(Vector3(10, 10, 10));
	//	std::cout << viewMatrix.GetPositionVector() << std::endl;
	glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "viewMatrix"), 1, false, (float*)&viewMatrix);

	glUniform1i(glGetUniformLocation(program, "diffuse_texture"), 0);
	glPointSize(6);
	glBindVertexArray(vao);
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


void Water::initParticles(float2 *p, int dx, int dy){
	int i, j;

	for (i = 0; i < dy; i++)
	{
		for (j = 0; j < dx; j++)
		{
			p[i*dx + j].x = (j + 0.5f + (MYRAND - 0.5f)) / dx;
			p[i*dx + j].y = (i + 0.5f + (MYRAND - 0.5f)) / dy;
		}
	}
}

void Water::simulateFluids(void)
{
	// simulate fluid
	advectVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM, DT);
	diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VIS);
	updateVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM);
	advectParticles(vbo, dvfield, DIM, DIM, DT);
}
void Water::addSomeForce(int spx, int spy, float fx, float fy) const{
	addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
}

