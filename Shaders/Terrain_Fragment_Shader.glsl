#version 330 core
layout(location = 0) out vec4 out_color;

uniform sampler2D diffuse_texture;

//in vec4 color;
//in vec2 texCoord;
void main()
{
	out_color = vec4(0,0,1,1);
	//out_color = texture(diffuse_texture, texCoord);
}