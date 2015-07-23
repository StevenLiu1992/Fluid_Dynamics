
#pragma once
# include <string>
# include <iostream>
# include <fstream>
#include <vector>
#include "Vector2.h"
#include "Model.h"

#define HEIGHTMAP_X 50
#define HEIGHTMAP_Y (1.f/500)
#define HEIGHTMAP_Z 50

#define HEIGHTMAP_TEX_X 1/256.f
#define HEIGHTMAP_TEX_Z 1/256.f

namespace Rendering
{
	namespace Models
	{
		class Terrain : public Models::Model
		{
		public:
			Terrain();
			~Terrain();

			void Create(std::string name, Core::Camera* c);
			virtual void Draw() override final;
			virtual void Update(Matrix4 viewMatrix) override final;
			void setTexture(GLuint t) { this->texture = t; }
			void GenerateNormals();
		private:
			int column;
			int row;
			float xllcorner;
			float yllcorner;
			float size;
			float NODATA_value;
			std::vector< std::vector<float>> heightmap;
			GLuint type;
			GLuint numVertices;
			GLuint numIndices;

			GLuint vbo_vertices;
			GLuint vbo_texturecord;
			GLuint vbo_normal;
			GLuint vbo_tangent;
			GLuint vbo_indices;


			Vector3 *vertices;
			Vector2 *textureCoords;
			unsigned int *indices;
			Vector3 *normals;
			Vector3 *tangents;
			Core::Camera* camera;
		};
	}
}