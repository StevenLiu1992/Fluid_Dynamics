#version 330 core


layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_velocity;


uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;


out Vertex	{
	vec4 velocity;
} OUT;

void main()
{
	OUT.velocity = in_velocity;
	gl_Position =  vec4(in_position, 1);

}