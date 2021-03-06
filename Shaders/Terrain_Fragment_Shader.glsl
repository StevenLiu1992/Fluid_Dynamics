#version 330 core
layout(location = 0) out vec4 out_color;

uniform vec3 cameraPos;
uniform sampler2D diffuse_texture;

in vec4 color;
in vec2 texCoord;
in vec3 normal;
in vec3 worldPos;
void main()
{
	
	vec3 lightPos = vec3(0,50,0);
	float lightRadius = 1000;
	vec4 lightColour =  vec4(1,1,1,1);
	vec4 diffuse = texture ( diffuse_texture , texCoord );
//	vec4 diffuse = vec4(0,0,1,0);
	vec3 incident = normalize( lightPos - worldPos );

	float lambert = max (0.0, dot ( incident,normal));
	float dist = length ( lightPos - worldPos);
	float atten = 1.0 - clamp ( dist / lightRadius , 0.0 , 1.0);

	vec3 viewDir = normalize ( cameraPos - worldPos);
	vec3 halfDir = normalize ( incident + viewDir);

	float rFactor = max (0.0 , dot( halfDir ,normal));
	float sFactor = pow ( rFactor , 50.0 );

	vec3 colour = ( diffuse.rgb * lightColour.rgb );
	colour += ( lightColour.rgb * sFactor ) * 0.33;
	out_color = vec4 ( colour * atten * lambert , diffuse.a);
	out_color.rgb += ( diffuse.rgb * lightColour.rgb ) * 0.4;
	
	
	
//	out_color = vec4(0,0,1,1);
	//out_color = texture(diffuse_texture, texCoord);
}