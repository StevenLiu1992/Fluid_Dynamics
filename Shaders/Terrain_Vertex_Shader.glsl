#version 330 core


layout(location = 0) in vec3 in_position;
//layout(location = 1) in vec2 in_texCoord;
//layout(location = 2) in vec3 in_normal;
//layout(location = 3) in vec3 in_tangent;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

//out vec4 color;
//out vec2 texCoord;

void main()
{
//	color = in_color;
//	texCoord = in_texCoord;
	gl_Position = (projMatrix * viewMatrix * modelMatrix) * vec4(in_position, 1);
//	gl_Position = modelMatrix * vec4(in_position, 1);
}