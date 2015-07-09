#version 150 core


layout(points) in;
layout(line_strip, max_vertices = 4) out;


uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

out vec4 color;

in Vertex
{
    vec4 velocity;
	float density;
}IN[];



void main()
{
	
	for(int i = 0; i < gl_in.length(); ++i){
		vec3 dir = IN[i].velocity.xyz;
		float amount = length(dir)*20;
		vec4 pos = gl_in[i].gl_Position;
		if(IN[i].velocity.w>0){
		mat4 mvp = projMatrix * viewMatrix * modelMatrix;
		color = vec4(0,1,0,1);
		gl_Position = mvp *pos;
		EmitVertex();
	//	color =  vec4(1,0,0,1);
		
		gl_Position = mvp * ( vec4(0.005 * normalize(dir),0)+pos);
		EmitVertex();
		color =  pos;
		gl_Position = mvp * ( vec4(0.005 * normalize(dir),0)+pos);
		EmitVertex();
		gl_Position = mvp * ( vec4(0.03 * normalize(dir),0)+pos);
		EmitVertex();
		
		EndPrimitive();
		}
		
		
		
		
		//color = vec4(1,1,0,1);
	//	gl_Position = mvp*(pos+IN[i].velocity);
	//gl_Position = pos + vec4(0.2,0.2,0.2,0.2);
		//EmitVertex();
	
	
	}
	
	
}