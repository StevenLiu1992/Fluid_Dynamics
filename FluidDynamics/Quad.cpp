#include "Quad.h"
using namespace Rendering;
using namespace Models;

Quad::Quad()
{
	position = Vector3(0, 0, 0);
	orientation = Quaternion::AxisAngleToQuaterion(Vector3(1, 0, 0), -90);
}

Quad::~Quad()
{}

void Quad::Create()
{
	GLuint vao;
	GLuint vbo;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	std::vector<VertexFormat> vertices;
	vertices.push_back(VertexFormat(glm::vec3(-1, -1, 0.0),//pos
		glm::vec4(1, 0, 0, 1), glm::vec2(0, 0)));   //color
	vertices.push_back(VertexFormat(glm::vec3(1, -1, 0.0),//pos
		glm::vec4(0, 0, 0, 1), glm::vec2(1, 0)));   //color
	vertices.push_back(VertexFormat(glm::vec3(-1, 1, 0.0),  //pos
		glm::vec4(0, 1, 0, 1), glm::vec2(0, 1)));   //color
	vertices.push_back(VertexFormat(glm::vec3(1, 1, 0.0),//pos
		glm::vec4(0, 0, 1, 1), glm::vec2(1, 1)));   //color
	//nothing different from Triangle model
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);            //here we have 4
	glBufferData(GL_ARRAY_BUFFER, sizeof(VertexFormat) * 4, &vertices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
		sizeof(VertexFormat), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE,
		sizeof(VertexFormat),
		(void*)(offsetof(VertexFormat, VertexFormat::color)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
		sizeof(VertexFormat),
		(void*)(offsetof(VertexFormat, VertexFormat::textureCoords)));
	glBindVertexArray(0);
	this->vao = vao;
	this->vbos.push_back(vbo);

	texture = SOIL_load_OGL_texture("../Textures/ground.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);
}

void Quad::Update(Matrix4 viewMatrix)
{
	
	Model::Update(viewMatrix);
}

void Quad::Draw()
{
	glUseProgram(program);
	glBindVertexArray(vao);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	Matrix4 modelMatrix = worldTransform*Matrix4::Scale(Vector3(10,10,10));

	//*Matrix4::Rotation(90,Vector3(1,0,0))
	//	std::cout << viewMatrix.GetPositionVector() << std::endl;
	glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "viewMatrix"), 1, false, (float*)&viewMatrix);

	glUniform1i(glGetUniformLocation(program, "diffuse_texture"), 0);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glUseProgram(0);
}