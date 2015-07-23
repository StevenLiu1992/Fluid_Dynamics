#include "Terrain.h"
using namespace Rendering;
using namespace Models;

Terrain::Terrain()
{
	position = Vector3(0, 5, 0);
	orientation = Quaternion();
}

Terrain::~Terrain()
{}

void Terrain::Create(std::string name, Core::Camera* c){

	camera = c;
	std::ifstream f(name.c_str(), std::ios::in);
	if (!f){
		std::cout << "cant open the file..." << std::endl;
		return;
	}
	int i=0, j=0;
	while (!f.eof()) {
		std::string currentLine;
		f >> currentLine;

		if (currentLine == "ncols") {
			f >> column;
		}
		else if (currentLine == "nrows"){
			f >> row;
			heightmap.resize(row, std::vector<float>(column, 0));
		}
		else if (currentLine == "xllcorner"){
			f >> xllcorner;
		}
		else if (currentLine == "yllcorner"){
			f >> yllcorner;
		}
		else if (currentLine == "cellsize"){
			f >> size;
		}
		else if(currentLine == "NODATA_value"){
			f >> NODATA_value;
			break;
		}
		else{}
	}
	while (!f.eof()){
		
			for (i = 0; i < column; i++){
				f >> heightmap[j][i];
				if (heightmap[j][i] == NODATA_value){
					heightmap[j][i] = 500;
				}
			}
			j++;
		
	}
	
	f.close();

	type = GL_TRIANGLES;
	
	heightmap.resize(row, std::vector<float>(column, 0));
	numVertices = row*column;
	numIndices = (row - 1)*(column - 1) * 6;
	vertices = new Vector3[numVertices];
	textureCoords = new Vector2[numVertices];
	indices = new GLuint[numIndices];
	normals = NULL;
	tangents = NULL;
//	textureCoords = NULL;

	for (int x = 0; x < row; ++x) {
		for (int z = 0; z < column; ++z) {
			int offset = (x * column) + z;
			vertices[offset] = Vector3(((float)x) / row, heightmap[x][z]*HEIGHTMAP_Y, ((float)z) / column);
			textureCoords[offset] = Vector2(x * HEIGHTMAP_TEX_X, z * HEIGHTMAP_TEX_Z);
		}
	}
//	memset(vertices, 0, numVertices*sizeof(Vector3));
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
	GenerateNormals();
//	numIndices = 3;
	glGenVertexArrays(1, &vao);
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
	vbos.push_back(vbo_vertices);
	if (indices) {
		glGenBuffers(1, &vbo_indices);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(GLuint),
			indices, GL_STATIC_DRAW);
		vbos.push_back(vbo_indices);
	}

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

	

	if (tangents) {
		glGenBuffers(1, &vbo_tangent);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_tangent);
		glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vector3), tangents, GL_STATIC_DRAW);
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(3);
	}
	glBindVertexArray(0);
	
	texture = SOIL_load_OGL_texture("../Textures/grass.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Terrain::Update(Matrix4 viewMatrix)
{

	Model::Update(viewMatrix);
}

void Terrain::Draw()
{
	glUseProgram(program);
	glBindVertexArray(vao);
	

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	Matrix4 modelMatrix = worldTransform*Matrix4::Scale(Vector3(10, 6, 10))*Matrix4::Translation(Vector3(0, -2, 0));

	//*Matrix4::Rotation(90,Vector3(1,0,0))
	//	std::cout << viewMatrix.GetPositionVector() << std::endl;
	glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, false, (float*)&modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(program, "viewMatrix"), 1, false, (float*)&viewMatrix);

	glUniform1i(glGetUniformLocation(program, "diffuse_texture"), 0);
//	Vector3 cameraPos = camera->GetPosition();
	glUniform3fv(glGetUniformLocation(program, "cameraPos"), 1, (float *)& camera->GetPosition());
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glCullFace(GL_BACK);
	glDrawElements(type, numIndices, GL_UNSIGNED_INT, 0);
	
	glBindVertexArray(0);
	glUseProgram(0);
}

void Terrain::GenerateNormals() {
	if (!normals) {
		normals = new Vector3[numVertices];
	}
	for (GLuint i = 0; i < numVertices; ++i){
		normals[i] = Vector3();
	}
	
	for (GLuint i = 0; i < numIndices; i += 3){
		unsigned int a = indices[i];
		unsigned int b = indices[i + 1];
		unsigned int c = indices[i + 2];

		Vector3 normal = Vector3::Cross((vertices[b] - vertices[a]), (vertices[c] - vertices[a]));

		normals[a] += normal;
		normals[b] += normal;
		normals[c] += normal;

	}

	for (GLuint i = 0; i < numVertices; ++i){
		normals[i].Normalise();
	}
}