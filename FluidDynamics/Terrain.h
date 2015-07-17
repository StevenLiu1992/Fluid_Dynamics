
#pragma once
# include <string>
# include <iostream>
# include <fstream>
#include <vector>
#include "Vector2.h"
#include "Model.h"

#define HEIGHTMAP_X 1
#define HEIGHTMAP_Y 1
#define HEIGHTMAP_Z 1

#define HEIGHTMAP_TEX_X 1/16.f
#define HEIGHTMAP_TEX_Z 1/16.f

namespace Rendering
{
	namespace Models
	{
		class Terrain : public Models::Model
		{
		public:
			Terrain();
			~Terrain();

			void Create(std::string name);
			virtual void Draw() override final;
			virtual void Update(Matrix4 viewMatrix) override final;
			void setTexture(GLuint t) { this->texture = t; }
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
		};
	}
}