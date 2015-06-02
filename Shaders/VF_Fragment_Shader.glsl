#version 330 core
layout(location = 0) out vec4 out_color;



in vec4 color;

void main()
{
	//out_color = vec4(color.x*color.w,color.y*color.w,color.z*color.w);
	out_color = color;
}