#include "Dependencies\glew\glew.h"
#include "Dependencies\freeglut\freeglut.h"
#include "Dependencies\glm\glm.hpp"
#include <iostream>

#include <stdio.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA helper functions
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "kernel.cuh"
#include "Shader_Manager.h";
#include "Init_GLUT.h"
#include "Scene_Manager.h"

extern "C" cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


int CUDAstuff(){
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	/*int fakeargc = 1;
	char *fakeargv[] = { "fake", NULL };
	cudaDeviceProp deviceProps;
	int devID;
	devID = findCudaGLDevice(fakeargc, (const char **)fakeargv);
	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
	printf("CUDA device [%s] has %d Multi-Processors\n",
		deviceProps.name, deviceProps.multiProcessorCount);*/
	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
}

using namespace Core;
using namespace Init;
using namespace Managers;




int main(int argc, char **argv){
	CUDAstuff();
	


	WindowInfo window(std::string("XXX"),
		50, 50,//position
		800, 600, //size
		true);//reshape

	ContextInfo context(3, 3, true);
	FramebufferInfo frameBufferInfo(true, true, true, true);
	Init_GLUT::init(window, context, frameBufferInfo);
	IListener* scene = new Managers::Scene_Manager();
	Init::Init_GLUT::setListener(scene);
	Init_GLUT::run();
//	system("PAUSE");
	delete scene;
	return 0;
}