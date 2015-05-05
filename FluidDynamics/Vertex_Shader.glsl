#version 330 core


layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;

uniform mat4 modelMatrix;

out vec4 color;

void main()
{
	color = in_color;
	gl_Position = modelMatrix*vec4(in_position, 1);
}