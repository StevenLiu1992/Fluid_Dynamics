#include "Water.h"
using namespace Rendering;
using namespace Models;


extern int window_width;
extern int window_height;
extern bool start_run = false;
extern float4 *dvfield = NULL;

float4 *hvfield = NULL;
float4 *dtemp = NULL;
float4 *dpressure = NULL;
float4 *ddivergence = NULL;
float *ddensity = NULL;
float *hdensity = NULL;
extern float4 *hintersection = NULL;
extern float3 *hnormal = NULL;
//level set function
float *dlsf = NULL;
float *hlsf = NULL;
float2 *dcontribution = NULL;

GLuint vbo3 = 0;                 // OpenGL vertex buffer object
GLuint vbo2 = 0;                 // OpenGL vertex buffer object
GLuint vbo1 = 0;                 // OpenGL vertex buffer object
GLuint vbo = 0;                 // OpenGL vertex buffer object
GLuint vbo_intersection = 0;
GLuint vbo_normal = 0;
struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
struct cudaGraphicsResource *cuda_vbo_resource1; // handles OpenGL-CUDA exchange
struct cudaGraphicsResource *cuda_vbo_resource2; // handles OpenGL-CUDA exchange
struct cudaGraphicsResource *cuda_vbo_intersection; // handles OpenGL-CUDA exchange

struct cudaGraphicsResource *textureCudaResource;
struct cudaGraphicsResource *cuda_vbo_normal;

// Texture pitch
size_t tPitch_v = 0;
size_t tPitch_t = 0;
size_t tPitch_d = 0;
size_t tPitch_p = 0;
size_t tPitch_den = 0; 
size_t tPitch_lsf = 0; 
size_t tPitch_ctb = 0;

extern "C"
void advect(float4 *v, int dx, int dy, int dz, float dt);
extern "C"
void diffuse(float4 *v, float4 *temp, int dx, int dy, int dz, float dt);
extern "C"
void projection(float4 *v, float4 *temp, float4 *pressure, float4* divergence, int dx, int dy, int dz, float dt);
extern "C"
void advectParticles(GLuint vbo, float4 *v, float *d, int dx, int dy, int dz, float dt);
extern "C"
void advectDensity(float4 *v, float *d, int dx, int dy, int dz, float dt);
extern "C"
void addForce(float4 *v, float *d, int dx, int dy, int dz, float dt);
extern "C"
void advectLevelSet(float4 *v, float *ls, int dx, int dy, int dz, float dt);
extern "C"
void correctLevelSet(float *ls, float2 *con, int dx, int dy, int dz, float dt);
extern "C"
void raycasting(int x, int y, float *ls, float3 camera);

Water::Water()
{
	ttt = 0;
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

void Water::Create(Core::Camera* c)
{
	
	particle_count = 0;
	camera = c;
	GLint bsize;
	tPitch_v = 0;
	tPitch_t = 0;
	tPitch_p = 0;
	tPitch_d = 0;
	tPitch_den = 0;
	tPitch_lsf = 0;
	tPitch_ctb = 0;

	texture = SOIL_load_OGL_texture("../Textures/water_particle.jpg",
		SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);
	int devID;
	cudaDeviceProp deviceProps;
	int fakeargc = 1;
	char *fakeargv[] = { "fake", NULL };
	devID = findCudaGLDevice(fakeargc, (const char **)fakeargv);

	// get number of SMs on this GPU
	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
	getLastCudaError("!!!GetDevice error");
	printf("CUDA device [%s] has %d Multi-Processors\n",
		deviceProps.name, deviceProps.multiProcessorCount);

	setupTexture();
	bindTexture();
	//Allocate host data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	hvfield		= (float4 *)malloc(sizeof(float4) * DS);
	hdensity	= (float *)malloc(sizeof(float) * DS);
	hlsf		= (float *)malloc(sizeof(float) * DS);
	particles	= (float3 *)malloc(sizeof(float3) * DS);

	//Allocate device data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	cudaMallocPitch((void **)&dvfield, &tPitch_v, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&dtemp, &tPitch_t, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&ddivergence, &tPitch_d, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&dpressure, &tPitch_p, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&ddensity, &tPitch_den, sizeof(float)*NX*NY, NZ);
	cudaMallocPitch((void **)&dlsf, &tPitch_lsf, sizeof(float)*NX*NY, NZ);
	cudaMallocPitch((void **)&dcontribution, &tPitch_ctb, sizeof(float2)*NX*NY, NZ);
	
	//initilize data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	initParticles_velocity(hvfield, dvfield);
	initLevelSetFunc(hlsf, dlsf);
	initParticles(particles, hlsf);
	init_density(hdensity, particles, ddensity);
	cudaMemset(dcontribution, 0, sizeof(float2)*NX*NY*NZ);

	//paritcles system vbo>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * DS, particles, GL_DYNAMIC_DRAW);
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);
	if (bsize != (sizeof(float3) * DS)) return;
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	this->vao = vao;
	this->vbos.push_back(vbo);

	//for velocity field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	v_position = (float3 *)malloc(sizeof(float3) * DS);
	initVelocityPosition(v_position, NX, NY, NZ);

	glGenVertexArrays(1, &grid_vao);
	glBindVertexArray(grid_vao);

	//velocity position
	glGenBuffers(1, &vbo1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo1);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * DS, v_position, GL_DYNAMIC_DRAW);
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);
	if (bsize != (sizeof(float3) * DS)) return;

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
	this->vbos.push_back(vbo1);

	//velocity value
	glGenBuffers(1, &vbo2);
	glBindBuffer(GL_ARRAY_BUFFER, vbo2);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * DS, hvfield, GL_DYNAMIC_DRAW);
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);
	if (bsize != (sizeof(float4) * DS)) return;
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);
	this->vbos.push_back(vbo2);

	//density value
	glGenBuffers(1, &vbo3);
	glBindBuffer(GL_ARRAY_BUFFER, vbo3);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)* DS, hdensity, GL_DYNAMIC_DRAW);
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);
	if (bsize != (sizeof(float)* DS)) return;
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
	this->vbos.push_back(vbo3);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//intersection visiable>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	glGenVertexArrays(1, &intersection_vao);
	glBindVertexArray(intersection_vao);
	hintersection = (float4 *)malloc(sizeof(float4) * 1024 * 1024);
	memset(hintersection, 0, sizeof(float4) * 1024 * 1024);
	glGenBuffers(1, &vbo_intersection);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_intersection);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * 1024 * 1024, hintersection, GL_DYNAMIC_DRAW);
	
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);
	this->vbos.push_back(vbo_intersection);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	hnormal = (float3 *)malloc(sizeof(float3) * 1024 * 1024);
	memset(hnormal, 0, sizeof(float3) * 1024 * 1024);
	glGenBuffers(1, &vbo_normal);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * 1024 * 1024, hnormal, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
	this->vbos.push_back(vbo_normal);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glBindVertexArray(0);

	
	//end of vbo and vao/////////////////////////////////////////////////////////////////////////////////////////////////////
	
	generateCube();
	generateFBO();
	//bind particle position vbo
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
	//bind velocity value vbo
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource1, vbo2, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
	//bind density value vbo
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource2, vbo3, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
	//bind intersection value vbo
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_intersection, vbo_intersection, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
	//bind normal value vbo
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_normal, vbo_normal, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

	int CUDAVersion ;
	cudaRuntimeGetVersion(&CUDAVersion);
	std::cout << "CUDA version: "<< CUDAVersion << std::endl;

	
	
}


void Water::Update(Matrix4 viewMatrix)
{
	if (start_run)
		simulateFluids();
	
	Model::Update(viewMatrix);
}

void Water::Draw()
{
	drawCube();
	Matrix4 modelMatrix;

	//draw particle field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	glUseProgram(program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	//*Matrix4::Scale(Vector3(10, 10, 10))
	modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(0, 1, 0));

	glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "viewMatrix"), 1, false, (float*)&viewMatrix);

	glUniform1i(glGetUniformLocation(program, "diffuse_texture"), 0);
	glPointSize(3);
	glBindVertexArray(vao);

	glDrawArrays(GL_POINTS, 0, DS);
	glUseProgram(0);

	//draw velocity field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	glUseProgram(program1);

	
	//*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(-1, 0, 0))
	modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(-1, 0, 0));

	glUniformMatrix4fv(glGetUniformLocation(program1, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program1, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program1, "viewMatrix"), 1, false, (float*)&viewMatrix);
	
	
//	std::cout << cameraPos << std::endl;
//	glPointSize(1);
	glBindVertexArray(grid_vao);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDrawArrays(GL_POINTS, 0, DS);
	glUseProgram(0);


	//draw density field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	glUseProgram(program2);


	//*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(-1, 0, 0))
	modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(1, 0, 0));

	glUniformMatrix4fv(glGetUniformLocation(program2, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program2, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program2, "viewMatrix"), 1, false, (float*)&viewMatrix);
	Vector3 cameraPos = camera->GetPosition();
	glUniform3f(glGetUniformLocation(program2, "gCameraPos"), cameraPos.x, cameraPos.y, cameraPos.z);
	//	std::cout << cameraPos << std::endl;
	//	glPointSize(1);
	glBindVertexArray(grid_vao);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDrawArrays(GL_POINTS, 0, DS);
	glUseProgram(0);

	//draw intersection>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	glUseProgram(intersection_program);


	//*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(-1, 0, 0))
	modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10));

	glUniformMatrix4fv(glGetUniformLocation(intersection_program, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(intersection_program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(intersection_program, "viewMatrix"), 1, false, (float*)&viewMatrix);
	cameraPos = camera->GetPosition();
	glUniform3f(glGetUniformLocation(intersection_program, "cameraPos"), cameraPos.x, cameraPos.y, cameraPos.z);
	//	std::cout << cameraPos << std::endl;
	//	glPointSize(1);
	glBindVertexArray(intersection_vao);
	glPointSize(3);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDrawArrays(GL_POINTS, 0, 1024 * 1024);
	glUseProgram(0);


	Matrix4 reverse_mv = Matrix4::Scale(Vector3(0.1, 0.1, 0.1));
	cameraPos = reverse_mv*cameraPos;
	raycasting(window_width, window_height, dlsf, make_float3(cameraPos.x, cameraPos.y, cameraPos.z));
}


#define MYRAND (rand() / (float)RAND_MAX)
#define MAX(a,b,c) (((a) > (b) ? (a) : (b)) > (c) ? ((a) > (b) ? (a) : (b)):(c))
#define MIN(a,b,c) (((a) < (b) ? (a) : (b)) < (c) ? ((a) < (b) ? (a) : (b)):(c))

void Water::initLevelSetFunc(float *h, float *d){
	int i, j, k;
	int count = 0;
	int num = 64;
	memset(h, -1, sizeof(float) * DS);
	for (k = 0; k < NZ; k++){
		for (i = 0; i < NY; i++){
			for (j = 0; j < NX; j++){
				if (j >= 12 && j <= 20 && i >= 12 && i <= 20 && (k == 12 || k == 20)){
					h[k*NX*NY + i*NX + j] = 0;
				}
				if (j >= 12 && j <= 20 && k >= 12 && k <= 20 && (i == 12 || i == 20)){
					h[k*NX*NY + i*NX + j] = 0;
				}
				if (i >= 12 && i <= 20 && k >= 12 && k <= 20 && (j == 12 || j == 20)){
					h[k*NX*NY + i*NX + j] = 0;
				}
				if (k < 12 || k > 20 || i < 12 || i > 20 || j < 12 || j > 20){
					int a = std::abs(j - 16) - 4;
					int b = std::abs(i - 16) - 4;
					int c = std::abs(k - 16) - 4;
					h[k*NX*NY + i*NX + j] = MAX(a, b, c);
				}
				if (k > 12 && k < 20 && i > 12 && i < 20 && j > 12 && j < 20){
					int a = 4 - std::abs(j - 16);
					int b = 4 - std::abs(i - 16);
					int c = 4 - std::abs(k - 16);
					h[k*NX*NY + i*NX + j] = - MIN(a, b, c);
				}
			}
		}
	}

	cudaMemcpy(d, h, sizeof(float)* DS, cudaMemcpyHostToDevice);
}

void Water::initParticles(float3 *p, float *l){
	int i, j, k;
	int count = 0;
	int num = 64;
	memset(p, 0, sizeof(float3) * DS);
	for (k = 0; k < NZ; k++){

		for (i = 0; i < NY; i++)
		{
			for (j = 0; j < NX; j++)
			{
				/*if (l[k*NX*NY + i*NX + j] == 0){
					for (int m = 0; m < num; m++){
						if (k == 12 || k == 20){

							p[count].x = (j + MYRAND - 0.5) / NX;
							p[count].y = (i + MYRAND - 0.5) / NY;
							p[count].z = (float)k / NZ;
							count++;
						}
						if (i == 12 || i == 20){

							p[count].x = (j + MYRAND - 0.5) / NX;
							p[count].y = (float)i / NY;
							p[count].z = (k + MYRAND - 0.5) / NZ;
							count++;
						}
						if (j == 12 || j == 20){

							p[count].x = (float)j / NX;
							p[count].y = (i + MYRAND - 0.5) / NY;
							p[count].z = (k + MYRAND - 0.5) / NZ;
							count++;
						}

					}
				}*/
				if (j >= 12 && j < 20 && i >= 12 && i < 20 && (k == 12 || k == 20)){
					for (int m = 0; m < num; m++){
						p[count].x = (float)(j + MYRAND) / NX;
						p[count].y = (float)(i + MYRAND) / NY;
						p[count].z = (float)k / NZ;
						count++;
					}
				}
				if (j >= 12 && j < 20 && k >= 12 && k < 20 && (i == 12 || i == 20)){
					for (int m = 0; m < num; m++){
						p[count].x = (float)(j + MYRAND) / NX;
						p[count].y = (float)i / NY;
						p[count].z = (float)(k + MYRAND) / NZ;
						count++;
					}
				}
				if (i >= 12 && i < 20 && k >= 12 && k < 20 && (j == 12 || j == 20)){
					for (int m = 0; m < num; m++){
						p[count].x = (float)j / NX;
						p[count].y = (float)(i + MYRAND) / NY;
						p[count].z = (float)(k + MYRAND) / NZ;
						count++;
					}
				}
			}
		}
	}
	particle_count = count;
	std::cout << "particle amount:" << particle_count << std::endl;
}
void Water::initVelocityPosition(float3 *vp, int dx, int dy, int dz){
	int i, j, k;
	for (k = 0; k <= (dz - 1); k++){

		for (i = 0; i <= (dy - 1); i++)
		{
			for (j = 0; j <= (dx - 1); j++)
			{
				vp[k*dx*dy + i*dx + j].x = (float)(j + 0.5) / dx;
				vp[k*dx*dy + i*dx + j].y = (float)(i + 0.5) / dy;
				vp[k*dx*dy + i*dx + j].z = (float)(k + 0.5) / dz;
			}
		}
	}
}

void Water::initParticles_velocity(float4 *h, float4 *d){
	int i, j, k;
	for (k = 0; k < NZ; k++){
		for (i = 0; i < NY; i++){
			for (j = 0; j < NX; j++){

				h[k*NX*NY + i*NX + j].x = 0;
				h[k*NX*NY + i*NX + j].y = 0;
				h[k*NX*NY + i*NX + j].z = 0;

				/*if (j>14 && j<18 && i>0 && i<12 && k>14 && k<18){
					h[k*NX*NY + i*NX + j].x = 0;
					h[k*NX*NY + i*NX + j].y = 0.2;
					h[k*NX*NY + i*NX + j].z = 0;
				}*/
				/*if (j>0 && j<10 && i>20 && i<26 && k>8 && k<28){
					h[k*NX*NY + i*NX + j].x = 0.7;
					h[k*NX*NY + i*NX + j].y = 0;
					h[k*NX*NY + i*NX + j].z = 0;
				}*/			
			}
		}
	}
	cudaMemcpy(d, h, sizeof(float4)* DS, cudaMemcpyHostToDevice);
}

void Water::init_density(float *h, float3* p, float *d){
	int i, j, k;
	float total = 0;
	memset(h, 0, sizeof(float) * DS);
	for (k = 1; k < NZ-1; k++){
		for (i = 1; i < NY-1; i++){
			for (j = 1; j < NX-1; j++){
				if (i >= 11 && i <= 21 && k >= 11 && k <= 21 && j >= 11 && j <= 21){
					h[k*NX*NY + i*NX + j] = 10.f;
					total += 10.f;
				}
			}
		}
	}
	std::cout <<"total density: "<< total << std::endl;
	cudaMemcpy(d, h, sizeof(float)* DS, cudaMemcpyHostToDevice);
}

void Water::cout_max_length_vector(float4* h){
	int i, j, k;
	int a, b, c, d, e, f;
	float max = 0, min = 10;
	/*for (k = 0; k <= (NZ - 1); k++){

		for (i = 0; i <= (NY - 1); i++)
		{
			for (j = 0; j <= (NX - 1); j++)
			{
				float sq = h[k*NZ*NY + i*NX + j].x*h[k*NZ*NY + i*NX + j].x +
					h[k*NZ*NY + i*NX + j].y*h[k*NZ*NY + i*NX + j].y +
					h[k*NZ*NY + i*NX + j].z*h[k*NZ*NY + i*NX + j].z;
				
				if (max < sq){
					a = j;
					b = i;
					c = k;
					max = sq;
				}
				if (min > sq){
					d = j;
					e = i;
					f = k;
					min = sq;
				}
			}
		}
	}
	std::cout << "time: "<<ttt << std::endl;
	std::cout << "max <" << h[c*NZ*NY + b*NX + a].x << "," << h[c*NZ*NY + b*NX + a].y << "," << h[c*NZ*NY + b*NX + a].z << ">" << std::endl;
	std::cout << "cor <" << a << "," << b << "," << c << ">" << std::endl;
	float qq = h[16 * NZ*NY + 22 * NX + 30].x*h[16 * NZ*NY + 22 * NX + 30].x + h[16 * NZ*NY + 22 * NX + 30].y*h[16 * NZ*NY + 22 * NX + 30].y + h[16 * NZ*NY + 22 * NX + 30].z*h[16 * NZ*NY + 22 * NX + 30].z;
	std::cout << qq << std::endl << std::endl;*/
//	std::cout << "cor <" << 30 << "," << 22 << "," << 16 << ">" << std::endl << std::endl;
//	std::cout << "cor <" << d << "," << e << "," << f << ">" << std::endl;
//	std::cout << "min <" << h[f*NZ*NY + e*NX + d].x << "," << h[f*NZ*NY + e*NX + d].y << "," << h[f*NZ*NY + e*NX + d].z << ">" << std::endl;
	/*k = 16;
	i = 16;
	for (j = 0; j < NX; j++){
		std::cout << h[k*NZ*NY + j*NX + i].x << "," << h[k*NZ*NY + j*NX + i].y << "," << h[k*NZ*NY + j*NX + i].z << std::endl;
	}*/
	std::cout << h[1*NZ*NY + 1*NX + 1].y;
	std::cout << h[1 * NZ*NY + 1 * NX + 1].y;
	std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
}
void Water::cout_density(float* d){
	int i, j, k;
	float total = 0;
	for (k = 0; k < NZ; k++){

		for (i = 0; i < NY; i++)
		{
			for (j = 0; j < NX; j++)
			{
			//	std::cout << d[k*NZ*NY + i*NX + j] << " ";	
				total += d[k*NZ*NY + i*NX + j];
			
			}
		
		}
	}
	std::cout<<"total density: " << total << std::endl;
	std::cout << d[1 * NZ*NY + 1 * NX + 1] << std::endl;
	std::cout << d[(NZ-2) * NZ*NY + 1 * NX + 1] << std::endl;
	std::cout << d[(NZ - 2) * NZ*NY + 1 * NX + NX-2] << std::endl;
	std::cout << d[1 * NZ*NY + 1 * NX + NX - 2] << std::endl;
	
}
void Water::cout_levelset(float* ls){
	int i, j, k;
	float total = 0;
	std::cout << "<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
//	for (k = 8; k < NZ-8; k++){
		for (i = NY-1; i >= 0; i--){
			for (j = 5; j < 30; j++){
			//	if (k == 16)
				if (ls[16 * NX*NY + i*NX + j] < 0 || ls[16 * NX*NY + i*NX + j] >= 10)
					printf("%1.f ", ls[16*NX*NY + i*NX + j]);
				else
					
					printf(" %1.f ", ls[16 * NX*NY + i*NX + j]);
					//std::cout << ls[k*NX*NY + i*NX + j] << " ";
			}
			std::cout << std::endl;
		}
//	}
	std::cout << "<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;

}
void Water::simulateFluids(void)
{
	// simulate fluid
	ttt++;
	addForce(dvfield, ddensity, NX, NY, NZ, DT);
	advect(dvfield, NX, NY, NZ, DT);
	diffuse(dvfield, dtemp, NX, NY, NZ, DT);
	projection(dvfield, dtemp, dpressure, ddivergence, NX, NY, NZ, DT);
	advectParticles(vbo, dvfield, ddensity, NX, NY, NZ, DT);
	advectDensity(dvfield, ddensity, NX, NY, NZ, DT);

	advectLevelSet(dvfield, dlsf, NX, NY, NZ, DT);
	correctLevelSet(dlsf, dcontribution, NX, NY, NZ, DT);


	//Matrix4 reverse_mv = Matrix4::Scale(Vector3(0.1, 0.1, 0.1));
	//Vector3 cameraPos = camera->GetPosition();
	//cameraPos = reverse_mv*cameraPos;
	//raycasting(window_width, window_height, dlsf, make_float3(cameraPos.x, cameraPos.y, cameraPos.z));

//	cudaMemcpy(hvfield, dvfield, sizeof(float4)* DS, cudaMemcpyDeviceToHost);
//	cout_max_length_vector(hvfield);
//	cudaMemcpy(hvfield, dpressure, sizeof(float4)* DS, cudaMemcpyDeviceToHost);
//	cout_max_length_vector(hvfield);
//	cudaMemcpy(hdensity, ddensity, sizeof(float)* DS, cudaMemcpyDeviceToHost);
//	cout_density(hdensity);
//	cudaMemcpy(hlsf, dlsf, sizeof(float)* DS, cudaMemcpyDeviceToHost);
//	cout_levelset(hlsf);
}

void Water::generateCube(){
	//cube data{position, colour, texturecord, index}
	std::vector<VertexFormat> vertices;
	vertices.push_back(VertexFormat(glm::vec3(0, 0, 1),
		glm::vec4(0, 0, 1, 1), glm::vec2(1, 0)));   
	vertices.push_back(VertexFormat(glm::vec3(1, 0, 1),
		glm::vec4(0, 0, 1, 1), glm::vec2(1, 1)));   
	vertices.push_back(VertexFormat(glm::vec3(1, 1, 1),  
		glm::vec4(1, 0, 0, 1), glm::vec2(0, 0)));   
	vertices.push_back(VertexFormat(glm::vec3(0, 1, 1),
		glm::vec4(1, 0, 0, 1), glm::vec2(0, 1)));   
	vertices.push_back(VertexFormat(glm::vec3(0, 0, 0),
		glm::vec4(0, 0, 1, 1), glm::vec2(1, 0)));   
	vertices.push_back(VertexFormat(glm::vec3(1, 0, 0),
		glm::vec4(0, 0, 1, 1), glm::vec2(0, 0)));   
	vertices.push_back(VertexFormat(glm::vec3(1, 1, 0),
		glm::vec4(1, 0, 0, 1), glm::vec2(0, 1)));   
	vertices.push_back(VertexFormat(glm::vec3(0, 1, 0),
		glm::vec4(1, 0, 0, 1), glm::vec2(1, 1)));

	GLuint indicesVboData[] = {
			0, 1, 2, 2, 3, 0,
			3, 2, 6, 6, 7, 3,
			7, 6, 5, 5, 4, 7,
			4, 0, 3, 3, 7, 4,
			0, 5, 1, 4, 5, 0,
			1, 5, 6, 6, 2, 1
	};

	glGenVertexArrays(1, &cube_vao);
	glBindVertexArray(cube_vao);
	
	glGenBuffers(1, &cube_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, cube_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VertexFormat) * 8, &vertices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
		sizeof(VertexFormat), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE,
		sizeof(VertexFormat),
		(void*)(offsetof(VertexFormat, VertexFormat::color)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
		sizeof(VertexFormat),
		(void*)(offsetof(VertexFormat, VertexFormat::textureCoords)));
	
	//add index data
	glGenBuffers(1, &cube_vbo_index);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_vbo_index);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 36 * sizeof(GLuint), indicesVboData, GL_STATIC_DRAW);

	glBindVertexArray(0);

//	this->vbos.push_back(cube_vbo);
}

void Water::drawCube(){
	glBindFramebuffer(GL_FRAMEBUFFER, cubeBufferFBO);
	glClearColor(0, 0, 0, 0);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	Matrix4 modelMatrix;

	//draw particle field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	glUseProgram(colorProgram);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	
	modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10));

	glUniformMatrix4fv(glGetUniformLocation(colorProgram, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(colorProgram, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(colorProgram, "viewMatrix"), 1, false, (float*)&viewMatrix);

	glUniform1i(glGetUniformLocation(colorProgram, "diffuse_texture"), 0);
	glBindVertexArray(cube_vao);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GenerateScreenTexture(GLuint & into, bool depth) {
	glGenTextures(1, &into);
	glBindTexture(GL_TEXTURE_2D, into);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	if (depth){
		glTexImage2D(GL_TEXTURE_2D, 0,
			GL_DEPTH_COMPONENT24,
			window_width, window_height, 0,
			 GL_DEPTH_COMPONENT,
			GL_UNSIGNED_BYTE, NULL);
	}
	else{
		glTexImage2D(GL_TEXTURE_2D, 0,
			GL_RGBA32F,
			window_width, window_height, 0,
			GL_RGBA,
			GL_FLOAT, NULL);
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	
}

void Water::generateFBO(){
	glGenFramebuffers(1, &cubeBufferFBO);
	
	GenerateScreenTexture(cubeDepthTexture, true);
	GenerateScreenTexture(cubePositionTexture, false);

	checkCudaErrors(cudaGraphicsGLRegisterImage(&textureCudaResource, cubePositionTexture, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegistertexture failed");

	//when we finished generate position texture, bind it to a cuda array (bind to cuda 2d texture)
	bindTexturetoCudaArray();
	
	// And now attach them to our FBO
	glBindFramebuffer(GL_FRAMEBUFFER, cubeBufferFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
		GL_TEXTURE_2D, cubePositionTexture, 0);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
		GL_TEXTURE_2D, cubeDepthTexture, 0);
	

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		return;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}