#pragma once
#include "Model.h"

namespace Rendering
{
	namespace Models
	{
		class Triangle : public Model
		{
		public:
			Triangle();
			~Triangle();

			void Create();
			virtual void Update(Matrix4 viewMatrix) override final;
			virtual void Draw() override final;
		};
	}
}