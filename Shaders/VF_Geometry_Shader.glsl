#version 150 core


layout(points) in;
layout(line_strip, max_vertices = 2) out;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

out vec4 color;

in Vertex
{
    vec4 velocity;
}IN[];



void main()
{
	
	for(int i = 0; i < gl_in.length(); ++i){
		
		vec4 pos = gl_in[i].gl_Position;
		if(pos.z>0.75&&pos.z<0.8){
		mat4 mvp = projMatrix * viewMatrix * modelMatrix;
		
		color = IN[i].velocity;
	//	color = vec4(1,0,0,1);
		gl_Position = mvp *pos;
		EmitVertex();
	
	
	//	color = vec4(0,0,1,1);
		
		gl_Position = mvp * (0.005* normalize(IN[i].velocity)+pos);
		EmitVertex();
		
		EndPrimitive();
		}
		
		
		
		
		//color = vec4(1,1,0,1);
	//	gl_Position = mvp*(pos+IN[i].velocity);
	//gl_Position = pos + vec4(0.2,0.2,0.2,0.2);
		//EmitVertex();
	
	
	}
	
	
}