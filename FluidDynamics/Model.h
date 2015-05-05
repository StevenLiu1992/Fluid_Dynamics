#pragma once
#include <vector>
#include "IGameObject.h"

namespace Rendering
{
	namespace Models{

		class Model :public IGameObject
		{
		public:
			Model();
			virtual ~Model();
			// methods from interface
			virtual void Draw() override;
			virtual void Update(Matrix4 viewMatrix) override;
			virtual void SetProgram(GLuint shaderName) override;
			virtual void Destroy() override;

			virtual GLuint GetVao() const override;
			virtual const std::vector<GLuint>& GetVbos() const override;

	//		void setViewMatrix(Matrix4 viewMatrix){ this->viewMatrix = viewMatrix; }
		protected:
			GLuint vao;
			GLuint program;
			std::vector<GLuint> vbos;
			Matrix4 worldTransform;
			Vector3 position;
			Quaternion orientation;
			Matrix4 viewMatrix;
		};
	}
}