//square/quad
#pragma once
#include "Model.h"
namespace Rendering
{
	namespace Models
	{
		class Cube : public Models::Model
		{
		public:
			Cube();
			~Cube();

			void Create();
			virtual void Draw() override final;
			virtual void Update(Matrix4 viewMatrix) override final;
			void setTexture(GLuint t) { this->texture = t; }
		};
	}
}