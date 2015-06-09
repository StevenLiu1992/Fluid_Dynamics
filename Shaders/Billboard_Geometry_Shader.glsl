#version 150 core


layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;


uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;
uniform vec3 gCameraPos;

out vec4 color;

in Vertex
{
    vec4 velocity;
	float density;
}IN[];



void main()
{
	
	for(int i = 0; i < gl_in.length(); ++i){
		vec3 pos = gl_in[i].gl_Position.xyz;
		vec3 worldpos = (modelMatrix*IN[i].velocity).xyz;
		mat4 mvp = projMatrix * viewMatrix * modelMatrix;
		color =  vec4(0,0,1,IN[i].density);
		gl_Position = mvp*vec4(pos+vec3(0,-0.03,0), 1.0);
		EmitVertex();
		gl_Position = mvp*vec4(pos+vec3(0.03,-0.03,0), 1.0);
		EmitVertex();
		gl_Position = mvp*vec4(pos+vec3(0,0,0), 1.0);
		EmitVertex();
		gl_Position = mvp *vec4(pos+vec3(0.03,0,0), 1.0);
		EmitVertex();
		EndPrimitive();
		/*
		vec3 toCamera = normalize(gCameraPos - worldpos);
		vec3 up = vec3(0.0, 1.0, 0.0);
		vec3 right = cross(toCamera, up);
		color =  vec4(0,0,1,IN[i].density*10);
		worldpos -= (right * 0.5);
		gl_Position = projMatrix * (viewMatrix *vec4(worldpos, 1.0));
		EmitVertex();

		worldpos.y += 1;
		gl_Position = projMatrix * (viewMatrix *vec4(worldpos, 1.0));
		EmitVertex();

		worldpos.y -= 1;
		worldpos += (right * 0.5);
		gl_Position = projMatrix * (viewMatrix * vec4(worldpos, 1.0));
		EmitVertex();

		worldpos.y += 1;
		gl_Position = projMatrix * (viewMatrix *vec4(worldpos, 1.0));
		EmitVertex();

		EndPrimitive();
		*/
	}
}