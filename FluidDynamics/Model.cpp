#include "Model.h"
using namespace Rendering;
using namespace Models;

Model::Model(){
	
	
	projMatrix = Matrix4::Perspective(1.0f, 10000.0f, (float)800 / (float)600, 45.0f);
}

Model::~Model()
{
	Destroy();
}

void Model::Draw()
{
	//this will be again overridden
}

void Model::Update(Matrix4 viewMatrix)
{
	//this will be again overridden
	worldTransform = orientation.ToMatrix();
	worldTransform.SetPositionVector(position);
	this->viewMatrix = viewMatrix;
}

void Model::SetProgram(GLuint program)
{
	this->program = program;
}

GLuint Model::GetVao() const
{
	return vao;
}

const std::vector<GLuint>& Model::GetVbos() const
{
	return vbos;
}

void Model::Destroy()
{
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(vbos.size(), &vbos[0]);
	vbos.clear();
}