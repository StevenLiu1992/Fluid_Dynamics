#version 330 core


layout(location = 0) in vec4 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texCoord;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

out vec4 color;
out vec3 normal;
out vec3 worldPos;
//out vec2 texCoord;

void main()
{
	
	//texCoord = in_texCoord;
	if(in_position.w>0){
		gl_Position = (projMatrix * viewMatrix * modelMatrix) * in_position;
		color = vec4(0,1,0,1);
		mat3 normalMatrix = transpose (inverse(mat3 ( modelMatrix )));

		normal = normalize ( normalMatrix * normalize ( in_normal ));
		worldPos = ( modelMatrix * in_position).xyz;
		
	}
	else{
		color = vec4(0,0,0,0);
	}
		
	
//	gl_Position = modelMatrix * vec4(in_position, 1);
}