#include "Camera.h"

/*
Polls the camera for keyboard / mouse movement.
Should be done once per frame! Pass it the msec since
last frame (default value is for simplicities sake...)
*/

void Core::Camera::keyboardEvents(char key){
	switch (key) {
	case 'w':{
		position += Matrix4::Rotation(yaw, Vector3(0, 1, 0)) * Vector3(0, 0, -1)*0.1 ;
		break;
	}
	
	case 's':{
		position -= Matrix4::Rotation(yaw, Vector3(0, 1, 0)) * Vector3(0, 0, -1)*0.1;
		break;
	}

	case 'a':{
		position += Matrix4::Rotation(yaw, Vector3(0, 1, 0)) * Vector3(-1, 0, 0) *0.1;
		break;
	}
	case 'd':{
		position -= Matrix4::Rotation(yaw, Vector3(0, 1, 0)) * Vector3(-1, 0, 0) *0.1;
		break;
	}
	case 'q':{
		position.y += 0.1;
		break;
	}
	case 'e':{
		position.y -= 0.1;
		break;
	}
	}
}
void Core::Camera::mouseMoveEvents(int delta_x, int delta_y){
	pitch -= delta_y * 0.07;
	yaw -= delta_x * 0.3;
	
	//Bounds check the pitch, to be between straight up and straight down ;)
	pitch = min(pitch,90.0f);
	pitch = max(pitch,-90.0f);
	
	if(yaw <0) {
		yaw+= 360.0f;
	}
	if(yaw > 360.0f) {
		yaw -= 360.0f;
	}
}
//void Camera::UpdateCamera(float msec)	{
//	this->msec = msec;
//	//Update the mouse by how much
//	pitch -= (Window::GetMouse()->GetRelativePosition().y);
//	yaw	  -= (Window::GetMouse()->GetRelativePosition().x);
//
//	//Bounds check the pitch, to be between straight up and straight down ;)
//	pitch = min(pitch,90.0f);
//	pitch = max(pitch,-90.0f);
//
//	if(yaw <0) {
//		yaw+= 360.0f;
//	}
//	if(yaw > 360.0f) {
//		yaw -= 360.0f;
//	}
//
//	msec /= 3.0f;
//
//	
//}

/*
Generates a view matrix for the camera's viewpoint. This matrix can be sent
straight to the shader...it's already an 'inverse camera' matrix.
*/
Matrix4 Core::Camera::BuildViewMatrix()	{
	//Why do a complicated matrix inversion, when we can just generate the matrix
	//using the negative values ;). The matrix multiplication order is important!
	return	Matrix4::Rotation(-pitch, Vector3(1,0,0)) * 
			Matrix4::Rotation(-yaw, Vector3(0,1,0)) * 
			Matrix4::Translation(-position);
};

Vector3 Core::Camera::BuildOrientation()	{
	Matrix4 m = BuildViewMatrix();
	
	return	Vector3(-m.values[2], -m.values[6], -m.values[10]);
};
