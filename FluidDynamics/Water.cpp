#include "Water.h"
using namespace Rendering;
using namespace Models;


extern int window_width;
extern int window_height;
extern bool start_run = false;
extern bool isAddSource = false;

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
int *hobstacle = NULL;
int *dobstacle = NULL;

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
size_t tPitch_obs = 0;

extern "C"
void exterapolation(float4 *v, float4 *temp, float *ls, int* obstacle);
extern "C"
void advect(float4 *v, float *l, int* obstacle);
//extern "C"
//void diffuse(float4 *v, float4 *temp, float *d);
extern "C"
void projection(float4 *v, float4 *temp, float4 *pressure, float4* divergence, float *d, int* obstacle);
extern "C"
void advectParticles(GLuint vbo, float4 *v, float *d);
extern "C"
void advectDensity(float4 *v, float *d);
extern "C"
void addForce(float4 *v, float *d, int* obstacle);
extern "C"
void advectLevelSet(float4 *v, float *ls);
extern "C"
void correctLevelSet(float *ls, float2 *con);
extern "C"
void raycasting(int x, int y, float *ls, float3 camera);
extern "C"
void addSource(float4 *v, float *d, float*l, int dx, int dy, int dz, float size);

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
	cudaFree(dobstacle);
	free(hobstacle);
	free(hdensity);
	free(hintersection);
	free(hnormal);
	free(hlsf);
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
	tPitch_obs = 0;

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
	hlsf		= (float *)malloc(sizeof(float) * LDS);
	particles	= (float3 *)malloc(sizeof(float3)* PAMOUNT);
	hobstacle   = (int *)malloc(sizeof(int) * DS);

	//Allocate device data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	cudaMallocPitch((void **)&dvfield, &tPitch_v, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&dtemp, &tPitch_t, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&ddivergence, &tPitch_d, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&dpressure, &tPitch_p, sizeof(float4)*NX*NY, NZ);
	cudaMallocPitch((void **)&ddensity, &tPitch_den, sizeof(float)*NX*NY, NZ);
	cudaMallocPitch((void **)&dobstacle, &tPitch_obs, sizeof(int)*NX*NY, NZ);
	
	cudaMallocPitch((void **)&dlsf, &tPitch_lsf, sizeof(float)*LNX*LNY, LNZ);
	cudaMallocPitch((void **)&dcontribution, &tPitch_ctb, sizeof(float2)*LNX*LNY, LNZ);
	//initilize data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	initParticles_velocity(hvfield, dvfield);
	initLevelSetFunc(hlsf, dlsf);
	initParticles(particles, hlsf);
	init_density(hdensity, particles, ddensity);
	init_obstacle(hobstacle, dobstacle);
	cudaMemset(dcontribution, 0, sizeof(float2)*LNX*LNY*LNZ);

	//paritcles system vbo>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)* PAMOUNT, particles, GL_DYNAMIC_DRAW);
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);
	if (bsize != (sizeof(float3)* PAMOUNT)) return;
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

	correctLevelSet(dlsf, dcontribution);
	
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

	glDrawArrays(GL_POINTS, 0, PAMOUNT);
	glUseProgram(0);

	//draw velocity field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//glUseProgram(program1);

	//
	////*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(-1, 0, 0))
	//modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(-1, 0, 0));

	//glUniformMatrix4fv(glGetUniformLocation(program1, "projMatrix"), 1, false, (float*)&projMatrix);
	//glUniformMatrix4fv(glGetUniformLocation(program1, "modelMatrix"), 1, false, (float*)&modelMatrix);
	//glUniformMatrix4fv(glGetUniformLocation(program1, "viewMatrix"), 1, false, (float*)&viewMatrix);
	//

	//glBindVertexArray(grid_vao);

	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//glDrawArrays(GL_POINTS, 0, DS);
	//glUseProgram(0);


	//draw density field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//glUseProgram(program2);


	////*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(-1, 0, 0))
	//modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 10, 10))*Matrix4::Translation(Vector3(1, 0, 0));

	//glUniformMatrix4fv(glGetUniformLocation(program2, "projMatrix"), 1, false, (float*)&projMatrix);
	//glUniformMatrix4fv(glGetUniformLocation(program2, "modelMatrix"), 1, false, (float*)&modelMatrix);
	//glUniformMatrix4fv(glGetUniformLocation(program2, "viewMatrix"), 1, false, (float*)&viewMatrix);
	//Vector3 cameraPos = camera->GetPosition();
	//glUniform3f(glGetUniformLocation(program2, "gCameraPos"), cameraPos.x, cameraPos.y, cameraPos.z);
	////	std::cout << cameraPos << std::endl;
	////	glPointSize(1);
	//glBindVertexArray(grid_vao);

	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glDisable(GL_CULL_FACE);
	//glDisable(GL_DEPTH_TEST);
	//glDrawArrays(GL_POINTS, 0, DS);
	//glUseProgram(0);

	//draw intersection>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	glUseProgram(intersection_program);
	Vector3 cameraPos = camera->GetPosition();

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
	glPointSize(2);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glDrawArrays(GL_POINTS, 0, 1024 * 1024);
	glUseProgram(0);


	Matrix4 reverse_mv = Matrix4::Scale(Vector3(0.1, 0.1, 0.1));
	cameraPos = reverse_mv*cameraPos;
	raycasting(window_width, window_height, dlsf, make_float3(cameraPos.x, cameraPos.y, cameraPos.z));
}


#define MYRAND (rand() / (float)RAND_MAX)
#define MAX(a,b,c) (((a) > (b) ? (a) : (b)) > (c) ? ((a) > (b) ? (a) : (b)):(c))
#define MIN(a,b,c) (((a) < (b) ? (a) : (b)) < (c) ? ((a) < (b) ? (a) : (b)):(c))

//int3 center = make_int3(16, 30, 20);
//int3 length = make_int3(12, 18, 18);

int3 center = make_int3(8, 15, 10);
int3 length = make_int3(6, 9, 9);

int3 start = make_int3(center.x - length.x, center.y - length.y, center.z - length.z);
int3 end = make_int3(center.x + length.x, center.y + length.y, center.z + length.z);

void Water::initLevelSetFunc(float *h, float *d){
	memset(h, 5, sizeof(float) * LDS);
	int i, j, k;


	for (k = 0; k < LNZ; k++){
		for (i = 0; i < LNY; i++){
			for (j = 0; j < LNX; j++){
				if (j >= start.x && j <= end.x && i >= start.y && i <= end.y && (k == start.z || k == end.z)){
					h[k*LNX*LNY + i*LNX + j] = 0;
				}
				if (j >= start.x && j <= end.x && k >= start.z && k <= end.z && (i == start.y || i == end.y)){
					h[k*LNX*LNY + i*LNX + j] = 0;
				}
				if (i >= start.y && i <= end.y && k >= start.z && k <= end.z && (j == start.x || j == end.x)){
					h[k*LNX*LNY + i*LNX + j] = 0;
				}
				if (k < start.z || k > end.z || i < start.y || i > end.y || j < start.x || j > end.x){
					int a = std::abs(j - center.x) - length.x;
					int b = std::abs(i - center.y) - length.y;
					int c = std::abs(k - center.z) - length.z;
					h[k*LNX*LNY + i*LNX + j] = MAX(a, b, c);
				}
				if (k > start.z && k < end.z && i > start.y && i < end.y && j > start.x && j < end.x){
					int a = length.x - std::abs(j - center.x);
					int b = length.y - std::abs(i - center.y);
					int c = length.z - std::abs(k - center.z);
					h[k*LNX*LNY + i*LNX + j] = -MIN(a, b, c);
				}
			/*	if (i > 3 && i < 24){
					h[k*LNX*LNY + i*LNX + j] = (i - 3) > (24 - i) ? (24 - i) : (i - 3);
				}
				if (i == 0){
					
					h[k*LNX*LNY + i*LNX + j] = -3;
				}
				if (i == 1){

					h[k*LNX*LNY + i*LNX + j] = -2;
				}
				if (i == 2){

					h[k*LNX*LNY + i*LNX + j] = -1;
				}
				if (i == 3){

					h[k*LNX*LNY + i*LNX + j] = 0;
				}*/
				
				/*if (j==2||j == 3||j==4){

					h[k*LNX*LNY + i*LNX + j] = -1;
				}*/
			}
		}
	}

	cudaMemcpy(d, h, sizeof(float)* LDS, cudaMemcpyHostToDevice);
}

void Water::initParticles(float3 *p, float *l){
	int i, j, k;
	int count = 0;
	int num = 64;
	memset(p, 0, sizeof(float3)* PAMOUNT);
	for (k = 0; k < NZ; k++){
		for (i = 0; i < NY; i++){
			for (j = 0; j < NX; j++){
				if (j >= start.x / TI && j < end.x / TI && 
					i >= start.y / TI && i < end.y / TI && 
					(k == start.z / TI || k == end.z / TI)){
					for (int m = 0; m < num; m++){
						p[count].x = (float)(j + MYRAND) / NX;
						p[count].y = (float)(i + MYRAND) / NY;
						p[count].z = (float)k / NZ;
						count++;
					}
				}
				if (j >= start.x / TI && j < end.x / TI &&
					k >= start.z / TI && k < end.z / TI &&
					(i == start.y / TI || i == end.y / TI)){
					for (int m = 0; m < num; m++){
						p[count].x = (float)(j + MYRAND) / NX;
						p[count].y = (float)i / NY;
						p[count].z = (float)(k + MYRAND) / NZ;
						count++;
					}
				}
				if (i >= start.y / TI && i < end.y / TI &&
					k >= start.z / TI && k < end.z / TI &&
					(j == start.x / TI || j == end.x / TI)){
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

void Water::init_obstacle(int *h, int*d){
	int i, j, k;
	memset(h, 0, sizeof(int) * DS);
	for (i = 0; i < NY; i++){
		for (j = 0; j < NX; j++){
			for (k = 0; k < NZ; k++){
				if (k == 0 || k == (NZ - 1) ||
					j == 0 || j == (NY - 1) ||
					i == 0 || i == (NX - 1)){
					h[i*NX*NY + j*NX + k] = 1;
				}
				/*if (k > 15 && k < 22 &&
					j > 0 && j < 30 &&
					i > 15 && i < 22){
					h[i*NX*NY + j*NX + k] = 1;
				}*/
				/*if (k>15 && i > 15){
					if (k > i){
						for (int xx = 0; xx < (i - 15); xx++){
							h[i*NX*NY + xx*NX + k] = 1;
						}
					}
					else{
						for (int xx = 0; xx < (k - 15); xx++){
							h[i*NX*NY + xx*NX + k] = 1;
						}
					}
				}*/

			}
		}
	}

	/*for (j = 0; j < NX; j++){
		for (k = 0; k < NZ; k++){ 
			
			std::cout << h[0*NX*NY + j*NX + k] << " ";
		}
		std::cout << std::endl;
	}*/
	cudaMemcpy(d, h, sizeof(int)* DS, cudaMemcpyHostToDevice);
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

			/*	if (j>14 && j<18 && i>0 && i<12 && k>14 && k<18){
					h[k*NX*NY + i*NX + j].x = 0;
					h[k*NX*NY + i*NX + j].y = 0.8;
					h[k*NX*NY + i*NX + j].z = 0;
				}
				if (j>0 && j<10 && i>20 && i<26 && k>8 && k<28){
					h[k*NX*NY + i*NX + j].x = 0.7;
					h[k*NX*NY + i*NX + j].y = 0;
					h[k*NX*NY + i*NX + j].z = 0;
				}	*/		
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
				if (i >= 12 && i <= 20 && k >= 12 && k <= 20 && j >= 12 && j <= 20){
					h[k*NX*NY + i*NX + j] = 10.f;
					total += 10.f;
				}
				/*if (i <= 1){
					h[k*NX*NY + i*NX + j] = 10.f;
					total += 10.f;
				}*/

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
	
	for (int i = 0; i < 32; i++){
		std::cout << i << " * "<<h[16 * NZ*NY + i * NX + 16].x << " " << h[16 * NZ*NY + i * NX + 16].y << " " << h[16 * NZ*NY + i * NX + 16].z << " " << h[16 * NZ*NY + i * NX + 16].w << std::endl;
	}
	
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


void Water::simulateFluids(void)
{
	// simulate fluid
	ttt++;
	if (isAddSource){
		addSource(dvfield, ddensity, dlsf, 5, 2, 5, 3);
		//isAddSource = false;
	}
	
	exterapolation(dvfield, dtemp, dlsf, dobstacle);
	advect(dvfield, dlsf, dobstacle);
	addForce(dvfield, dlsf, dobstacle);
//	diffuse(dvfield, dtemp, dlsf);
	projection(dvfield, dtemp, dpressure, ddivergence, dlsf, dobstacle);
	advectParticles(vbo, dvfield, ddensity);
//	advectDensity(dvfield, ddensity);

	advectLevelSet(dvfield, dlsf);
	correctLevelSet(dlsf, dcontribution);

	//	cudaMemcpy(hvfield, dvfield, sizeof(float4)* DS, cudaMemcpyDeviceToHost);
	//	cout_max_length_vector(hvfield);
	//	cudaMemcpy(hvfield, dpressure, sizeof(float4)* DS, cudaMemcpyDeviceToHost);
	//	cout_max_length_vector(hvfield);
	//	cudaMemcpy(hdensity, ddensity, sizeof(float)* DS, cudaMemcpyDeviceToHost);
	//	cout_density(hdensity);
	//	cudaMemcpy(hlsf, dlsf, sizeof(float)* LDS, cudaMemcpyDeviceToHost);
	//	cout_levelset(hlsf);
}
void Water::cout_levelset(float* ls){

	std::cout << "<<<<<<<<<<<<<<<levelset>>>>>>>>>>>>>>>>>" << std::endl;
	
	for (int i = 0; i < 64; i++){
		std::cout << i << " * " << ls[32 * LNZ*LNY + i * LNX + 32] << std::endl;
	}
	std::cout << "<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>" << std::endl;

}