#ifndef  VertexFormat_H_
#define VertexFormat_H_

#include "Dependencies\glm\glm.hpp" //installed with NuGet
namespace Rendering{

	struct VertexFormat
	{

		glm::vec3 position;
		glm::vec4 color;

		VertexFormat(const glm::vec3 &iPos, const glm::vec4 &iColor)
		{
			position = iPos;
			color = iColor;
		}

	};
}

#endif