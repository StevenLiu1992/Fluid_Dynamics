#version 330 core
layout(location = 0) out vec4 out_color;

uniform sampler2D diffuse_texture;

in vec4 color;
in vec2 texCoord;
void main()
{
	out_color = color;
	//out_color = texture(diffuse_texture, texCoord);
}