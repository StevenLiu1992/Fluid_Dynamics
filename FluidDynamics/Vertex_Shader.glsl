#version 330 core


layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

out vec4 color;

void main()
{
	color = in_color;
	gl_Position = (projMatrix * viewMatrix * modelMatrix) * vec4(in_position, 1);
//	gl_Position = modelMatrix * vec4(in_position, 1);
}