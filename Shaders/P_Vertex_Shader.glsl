#version 330 core


layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;
//layout(location = 2) in vec2 in_texCoord;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

out vec4 color;
out vec2 texCoord;

void main()
{
	color = vec4(in_position,1);
//	texCoord = in_texCoord;
	
	gl_Position = (projMatrix * viewMatrix * modelMatrix) * vec4(in_position, 1);
//	gl_Position = modelMatrix * vec4(in_position, 1);
}