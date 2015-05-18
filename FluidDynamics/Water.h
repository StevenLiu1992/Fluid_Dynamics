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

			void Create();
			virtual void Update(Matrix4 viewMatrix) override final;
			virtual void Draw() override final;
			void initParticles(float3 *p, int dx, int dy, int dz);
			void simulateFluids(void);
			void addSomeForce(int spx, int spy, float fx, float fy) const;
		private:
			
			

			
			GLuint vao;
			float3 *particles;
		
		};
		
	}
}