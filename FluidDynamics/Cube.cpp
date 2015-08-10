#include "Cube.h"
using namespace Rendering;
using namespace Models;

Cube::Cube()
{
	position = Vector3(0, 0, 0);
	orientation = Quaternion();
}

Cube::~Cube()
{}

void Cube::Create()
{
	GLuint vao;
	GLuint vbo;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);


	std::vector<VertexFormat> vertices;
	vertices.push_back(VertexFormat(glm::vec3(0, 0, 1),
		glm::vec4(0, 0, 1, 1), glm::vec2(0, 0)));
	vertices.push_back(VertexFormat(glm::vec3(1, 0, 1),
		glm::vec4(0, 0, 1, 1), glm::vec2(1, 0)));
	vertices.push_back(VertexFormat(glm::vec3(1, 1, 1),
		glm::vec4(1, 0, 0, 1), glm::vec2(1, 1)));
	vertices.push_back(VertexFormat(glm::vec3(0, 1, 1),
		glm::vec4(1, 0, 0, 1), glm::vec2(0, 1)));
	vertices.push_back(VertexFormat(glm::vec3(0, 0, 0),
		glm::vec4(0, 0, 1, 1), glm::vec2(1, 0)));
	vertices.push_back(VertexFormat(glm::vec3(1, 0, 0),
		glm::vec4(0, 0, 1, 1), glm::vec2(0, 0)));
	vertices.push_back(VertexFormat(glm::vec3(1, 1, 0),
		glm::vec4(1, 0, 0, 1), glm::vec2(0, 1)));
	vertices.push_back(VertexFormat(glm::vec3(0, 1, 0),
		glm::vec4(1, 0, 0, 1), glm::vec2(1, 1)));

	GLuint indicesVboData[] = {
		0, 1, 2, 2, 3, 0,
		3, 2, 6, 6, 7, 3,
		7, 6, 5, 5, 4, 7,
		4, 0, 3, 3, 7, 4,
		0, 5, 1, 4, 5, 0,
		1, 5, 6, 6, 2, 1
	};

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VertexFormat) * 8, &vertices[0], GL_STATIC_DRAW);
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

	GLuint cube_vbo_index;
	//add index data
	glGenBuffers(1, &cube_vbo_index);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_vbo_index);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 36 * sizeof(GLuint), indicesVboData, GL_STATIC_DRAW);

	glBindVertexArray(0);


	this->vao = vao;
	this->vbos.push_back(vbo);

	texture = SOIL_load_OGL_texture("../Textures/ncl.png", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);
	texture1 = SOIL_load_OGL_texture("../Textures/openglogo.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);
}

void Cube::Update(Matrix4 viewMatrix)
{

	Model::Update(viewMatrix);
}

void Cube::Draw()
{
	glUseProgram(program);
	glBindVertexArray(vao);

	glActiveTexture(GL_TEXTURE0);
	

	Matrix4 modelMatrix;

	//*Matrix4::Rotation(90,Vector3(1,0,0))
	//	std::cout << viewMatrix.GetPositionVector() << std::endl;
	glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "viewMatrix"), 1, false, (float*)&viewMatrix);
	glUniform1i(glGetUniformLocation(program, "diffuse_texture"), 0);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);
	glBindTexture(GL_TEXTURE_2D, texture1);
	
	//wall back/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	glUniform1i(glGetUniformLocation(program, "iswall"), 1);

	modelMatrix = worldTransform*Matrix4::Translation(Vector3(0.5, 0, 0))*Matrix4::Scale(Vector3(9.5, 4, 0.1));
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

	//wall right//////////////////////////////////////////////////////////////////////////////////////////////////////////
	modelMatrix = worldTransform*Matrix4::Translation(Vector3(9.5, 0, 0))*Matrix4::Scale(Vector3(0.5, 4, 9.5));
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	//wall front//////////////////////////////////////////////////////////////////////////////////////////////////////////
	modelMatrix = worldTransform*Matrix4::Translation(Vector3(0, 0, 9.5))*Matrix4::Scale(Vector3(9.5, 4, 0.1));
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	//wall left//////////////////////////////////////////////////////////////////////////////////////////////////////////
	modelMatrix = worldTransform*Matrix4::Translation(Vector3(0, 0, 0))*Matrix4::Scale(Vector3(0.5, 4, 9.5));
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	//wall bottom//////////////////////////////////////////////////////////////////////////////////////////////////////////
	modelMatrix = worldTransform*Matrix4::Translation(Vector3(0, 0, 0))*Matrix4::Scale(Vector3(10, 0.1, 9.5));
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glUniform1i(glGetUniformLocation(program, "iswall"), 0);
	modelMatrix = worldTransform*Matrix4::Translation(Vector3(4.7, 0, 4.7))*Matrix4::Scale(Vector3(1.4, 2.8, 1.4));
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	glBindTexture(GL_TEXTURE_2D, texture);
	modelMatrix = worldTransform*Matrix4::Translation(Vector3(4.7, 2.8, 4.7))*Matrix4::Scale(Vector3(1.4, 1.4, 1.4));
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	


	glBindVertexArray(0);
	glUseProgram(0);
}