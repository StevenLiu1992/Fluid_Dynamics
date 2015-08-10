#version 330 core
layout(location = 0) out vec4 out_color;

uniform sampler2D diffuse_texture;
uniform int iswall;

in vec4 color;
in vec2 texCoord;
void main()
{
	//out_color = color;
	if(iswall==1){
	out_color = vec4(0.8,0.8,0.9,0.1);
	//	out_color = texture(diffuse_texture, texCoord);
	//	out_color.a = 0.7;
	}
	else
		out_color = texture(diffuse_texture, texCoord);
}