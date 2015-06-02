#version 330 core


layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_velocity;
layout(location = 2) in float in_density;


uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;


out Vertex	{
	vec4 velocity;
	float density;
} OUT;

void main()
{
	OUT.density = in_density;
	OUT.velocity = in_velocity;
	gl_Position =  vec4(in_position, 1);

}