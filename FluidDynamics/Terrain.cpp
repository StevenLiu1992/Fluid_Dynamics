#include "Terrain.h"
using namespace Rendering;
using namespace Models;

Terrain::Terrain()
{
	position = Vector3(0, 0, 0);
	orientation = Quaternion::AxisAngleToQuaterion(Vector3(1, 0, 0), 90);
}

Terrain::~Terrain()
{}

void Terrain::Create(std::string name)
{
	std::ifstream f(name.c_str(), std::ios::in);
	if (!f){
		std::cout << "cant open the file..." << std::endl;
		return;
	}
	int i=0, j=0;
	//while (!f.eof()) {
	//	std::string currentLine;
	//	f >> currentLine;

	//	if (currentLine == "ncols") {
	//		f >> column;
	//	}
	//	else if (currentLine == "nrows"){
	//		f >> row;
	//		heightmap.resize(row, std::vector<float>(column, 0));
	//	}
	//	else if (currentLine == "xllcorner"){
	//		f >> xllcorner;
	//	}
	//	else if (currentLine == "yllcorner"){
	//		f >> yllcorner;
	//	}
	//	else if (currentLine == "cellsize"){
	//		f >> size;
	//	}
	//	else if (currentLine == "NODATA_value"){
	//		f >> NODATA_value;
	//	}
	//	else{
	//		/*for (i = 0; i < column; i++){
	//			f >> heightmap[j][i];
	//		}
	//		j++;*/
	//	}
	//}
	//for (j = 0; j < row; j++){
	//	for (i = 0; i < column; i++){
	//		if (heightmap[j][i] != -9999 && (heightmap[j][i]<0 || heightmap[j][i]>500)){
	//			std::cout << "!";
	//		}
	//	}
	//}
	f.close();

	type = GL_TRIANGLES;
	row = 1850;
	column = 2460;
	heightmap.resize(row, std::vector<float>(column, 0));
	numVertices = row*column;
	numIndices = (row - 1)*(column - 1) * 6;
	Vector3 *vertices = new Vector3[numVertices];
	Vector2 *textureCoords;// = new Vector2[numVertices];
	GLuint  *indices = new GLuint[numIndices];
	Vector3 *normals = NULL;
	Vector3 *tangents = NULL;
	textureCoords = NULL;

	for (int x = 0; x < row; ++x) {
		for (int z = 0; z < column; ++z) {
			int offset = (x * column) + z;

			vertices[offset] = Vector3(((float)x) / row * HEIGHTMAP_X, heightmap[x][z] * HEIGHTMAP_Y, ((float)z) / column * HEIGHTMAP_Z);
		//	setVertexColour(data[offset], offset);
		//	textureCoords[offset] = Vector2(x * HEIGHTMAP_TEX_X, z * HEIGHTMAP_TEX_Z);
		}
	}
	memset(vertices, 0, numVertices*sizeof(Vector3));
	numIndices = 0;

	for (int x = 0; x < row - 1; ++x) {
		for (int z = 0; z < column - 1; ++z) {
			int a = (x * (column)) + z;
			int b = ((x + 1) * (column)) + z;
			int c = ((x + 1) * (column)) + (z + 1);
			int d = (x * (column)) + (z + 1);

			indices[numIndices++] = c;
			indices[numIndices++] = b;
			indices[numIndices++] = a;

			indices[numIndices++] = c;
			indices[numIndices++] = a;
			indices[numIndices++] = d;

		}

	}

//	numIndices = 3;

	glBindVertexArray(vao);
	glGenBuffers(1, &vbo_vertices);// Generate 1 buffer, put the resulting identifier in bufferObject[vertexbuffer]
	glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);//start do something in the 'vertexbuffer'
	glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vector3),
		vertices, GL_DYNAMIC_DRAW);// Give our vertices to OpenGL
	glVertexAttribPointer(
		0,		// attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0			// array buffer offset
		);
	glEnableVertexAttribArray(0);

	if (textureCoords) { // This bit is new !
		glGenBuffers(1, &vbo_texturecord);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_texturecord);
		glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vector2),
			textureCoords, GL_STATIC_DRAW);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);



	}
	if (normals){
		glGenBuffers(1, &vbo_normal);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
		glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vector3),
			normals, GL_STATIC_DRAW);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);
	}

	if (indices) {
		glGenBuffers(1, &vbo_indices);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(GLuint),
			indices, GL_STATIC_DRAW);
	}

	if (tangents) {
		glGenBuffers(1, &vbo_tangent);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_tangent);
		glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vector3), tangents, GL_STATIC_DRAW);
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(3);
	}
	glBindVertexArray(0);
	
	texture = SOIL_load_OGL_texture("../Textures/ground.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);
}

void Terrain::Update(Matrix4 viewMatrix)
{

	Model::Update(viewMatrix);
}

void Terrain::Draw()
{
	glUseProgram(program);
	glBindVertexArray(vao);

	//glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, texture);

	Matrix4 modelMatrix = worldTransform;

	//*Matrix4::Rotation(90,Vector3(1,0,0))
	//	std::cout << viewMatrix.GetPositionVector() << std::endl;
	glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "viewMatrix"), 1, false, (float*)&viewMatrix);

	glUniform1i(glGetUniformLocation(program, "diffuse_texture"), 0);

	glDrawElements(type, numIndices, GL_UNSIGNED_INT, 0);
	
	glBindVertexArray(0);
	glUseProgram(0);
}