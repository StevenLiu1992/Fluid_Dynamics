#version 330 core
layout(location = 0) out vec4 out_color;

uniform vec3 cameraPos;

in vec4 color1;
in vec3 normal1;
in vec3 worldPos1;


void main()
{
	if(color1.a == 0)
		discard;
	
	vec3 lightPos = vec3(50,20,50);
	vec4 diffuse = vec4(0.3,0.4,0.9,1);
	float lightRadius = 100;
	vec4 lightColour = vec4(1,1,1,1);
	
	vec3 incident = normalize( lightPos - worldPos1 );
	float lambert = max (0.0, dot ( incident, normal1));
	float dist = length ( lightPos - worldPos1);
	float atten = 1.0 - clamp ( dist / lightRadius , 0.0 , 1.0);

	vec3 viewDir = normalize ( cameraPos - worldPos1);
	vec3 halfDir = normalize ( incident + viewDir);

	float rFactor = max (0.0 , dot( halfDir ,normal1));
	float sFactor = pow ( rFactor , 50.0 );

	vec3 colour = ( diffuse.rgb * lightColour.rgb );
	colour += ( lightColour.rgb * sFactor ) * 0.8;
	out_color = vec4 ( colour * atten * lambert , diffuse.a);
	out_color.rgb += ( diffuse.rgb * lightColour.rgb ) * 0.4;
	out_color.a = 0.9;
}