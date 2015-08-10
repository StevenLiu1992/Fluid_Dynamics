#version 150 core


layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;


uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

uniform vec3 cameraPos;



in Vertex
{
    vec4 color;
	vec3 normal;
	vec3 worldPos;
}IN[];

out vec4 color1;
out vec3 normal1;
out vec3 worldPos1;

void main()
{
	float size = 0.03;
	for(int i = 0; i < gl_in.length(); ++i){
		color1 = IN[i].color;
		normal1 = IN[i].normal;
		worldPos1 = IN[i].worldPos;
		
		vec3 pos = gl_in[i].gl_Position.xyz;
		vec3 toCamera = normalize(cameraPos - IN[i].worldPos);
		vec3 up = vec3(0.0, 1.0, 0.0);
		vec3 right = cross(toCamera, up);
		
		mat4 vp = projMatrix * viewMatrix;

		gl_Position = vp*vec4(IN[i].worldPos+vec3(0,-size,0), 1.0);
		EmitVertex();
		gl_Position = vp*vec4(IN[i].worldPos+vec3(0,-size,0)+right*size, 1.0);
		EmitVertex();
		gl_Position = vp*vec4(IN[i].worldPos+vec3(0,0,0), 1.0);
		EmitVertex();
		gl_Position = vp *vec4(IN[i].worldPos+right*size, 1.0);
		EmitVertex();
		EndPrimitive();
		
	}
}