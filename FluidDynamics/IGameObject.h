#pragma once
#include <vector>
#include <iostream>
#include "Dependencies/glew/glew.h"
#include "Dependencies/freeglut/freeglut.h"
#include "Dependencies/soil/soil.h"
#include "VertexFormat.h"
#include "Matrix4.h"
#include "Vector3.h"
#include "Quaternion.h"


namespace Rendering
{
	class IGameObject
	{
	public:
		virtual ~IGameObject() = 0;

		virtual void Draw() = 0;
		virtual void Update(Matrix4 viewMatrix) = 0;
		virtual void SetProgram(GLuint shaderName) = 0;
		virtual void Destroy() = 0;
		virtual GLuint GetVao() const = 0;
		virtual const std::vector<GLuint>& GetVbos() const = 0;
		
	};

	inline IGameObject::~IGameObject()
	{//blank
	}
}
