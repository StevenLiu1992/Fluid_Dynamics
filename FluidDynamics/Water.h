#pragma once

#include "WaterDefine.h"

#include "Model.h"
#include "Vector2.h"

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

// CUDA helper functions
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "WaterKernel.cuh"

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
			void initParticles(float2 * particles, int, int);
			void simulateFluids(void);
			void addSomeForce(int spx, int spy, float fx, float fy) const;
		private:
			
			

			
			GLuint vao;
			float2 *particles;
		
		};
		
	}
}