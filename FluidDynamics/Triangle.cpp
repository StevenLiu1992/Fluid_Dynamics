#include "Triangle.h"
using namespace Rendering;
using namespace Models;


Triangle::Triangle()
{
	position = Vector3(0, 0, 0);
}

Triangle::~Triangle()
{
	//is going to be deleted in Models.cpp (inheritance)
}

void Triangle::Create()
{
	//this is just copy past from
	// http://in2gpu.com/2014/12/19/create-triangle-opengl-part-iii-attributes/
	GLuint vao;
	GLuint vbo;

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	std::vector<VertexFormat> vertices;
	vertices.push_back(VertexFormat(glm::vec3(0, 0, 0.0),
		glm::vec4(1, 0, 0, 1), glm::vec2(0, 0)));
	vertices.push_back(VertexFormat(glm::vec3(1, 0, 0.0),
		glm::vec4(0, 1, 0, 1), glm::vec2(1, 0)));
	vertices.push_back(VertexFormat(glm::vec3(0, 1, 0.0),
		glm::vec4(0, 0, 1, 1), glm::vec2(1, 1)));

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VertexFormat)* 3, &vertices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFormat),
		(void*)0);
	glEnableVertexAttribArray(1);
	// you can use offsetof to get the offset of an attribute
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexFormat),
		(void*)(offsetof(VertexFormat, VertexFormat::color)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexFormat),
		(void*)(offsetof(VertexFormat, VertexFormat::textureCoords)));
	glBindVertexArray(0);

	//here we assign the values
	this->vao = vao;
	this->vbos.push_back(vbo);
	texture = SOIL_load_OGL_texture("../Textures/Background.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);
}

void Triangle::Update(Matrix4 viewMatrix)
{
	//for triangle there is nothing to update for now
	
//	this->viewMatrix = viewMatrix;
	Model::Update(viewMatrix);
}

void Triangle::Draw()
{
	glUseProgram(program);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	
	Matrix4 modelMatrix = worldTransform; 
//	modelMatrix.SetScalingVector(Vector3(10, 10, 10));
//	std::cout << viewMatrix.GetPositionVector() << std::endl;
	glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "viewMatrix"), 1, false, (float*)&viewMatrix);

	glUniform1i(glGetUniformLocation(program, "diffuse_texture"), 0);

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glUseProgram(0);
}