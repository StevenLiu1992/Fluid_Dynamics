#pragma once

#include "fluidDefine.h"

#include "Model.h"


// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>



// CUDA helper functions
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "fluidKernel.cuh"
namespace Rendering
{
	namespace Models
	{
		class Water : public Model
		{
		public:
			Water();
			~Water();

			void Create(Core::Camera* c = NULL);
			virtual void Update(Matrix4 viewMatrix) override final;
			virtual void Draw() override final;
			void initParticles(float3 *p, float *l);
			void initLevelSetFunc(float *h, float *d);
			void initParticles_velocity(float4 *v, float4 *d);
			void simulateFluids(void);
			void addSomeForce(int spx, int spy, float fx, float fy) const;
			void initVelocityPosition(float3 *vp, int dx, int dy, int dz);
			void SetProgram1(GLuint p1){ this->program1 = p1; }
			void SetProgram2(GLuint p2){ this->program2 = p2; }
			void SetColorProgram(GLuint p) { this->colorProgram = p; }
			void SetInterProgram(GLuint p) { this->intersection_program = p; }
			void cout_max_length_vector(float4* h);
			void init_density(float *h, float3* p, float *d);
			void init_obstacle(int *h, int*d);
			void cout_density(float* d);
			void cout_levelset(float* ls);
			void generateCube();
			void drawCube();
			void generateFBO();
			GLuint getTexture(){ return this->cubePositionTexture; }
		private:
			
			
			Core::Camera* camera;
			
			GLuint vao;
			GLuint grid_vao;
			GLuint intersection_vao;
			float3 *particles;
			float3 * v_position;
			GLuint program1;
			GLuint program2;
			GLuint intersection_program;
			int particle_count;
			int ttt;
			

			GLuint cube_vao;
			GLuint cube_vbo;
			GLuint cube_vbo_index;
			GLuint colorProgram;
			GLuint cubeBufferFBO;
			GLuint cubePositionTexture;
			GLuint cubeDepthTexture;
		};
		
	}
}