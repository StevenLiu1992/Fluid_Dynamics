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
			void initParticles(float3 *p, int dx, int dy, int dz);
			void initParticles_velocity(float4 *v, float4 *d);
			void simulateFluids(void);
			void addSomeForce(int spx, int spy, float fx, float fy) const;
			void initVelocityPosition(float3 *vp, int dx, int dy, int dz);
			void SetProgram1(GLuint p1){ this->program1 = p1; }
			void cout_max_length_vector(float4* h);
			void init_density(float *h, float3* p, float *d);
			void cout_density(float* d);
			
		private:
			
			
			Core::Camera* camera;
			
			GLuint vao;
			float3 *particles;
			float3 * v_position;
			GLuint program1;
			int ttt;
		};
		
	}
}