/******************************************************************************
Class:Camera
Implements:
Author:Rich Davison	<richard.davison4@newcastle.ac.uk>
Description:FPS-Style camera. Uses the mouse and keyboard from the Window
class to get movement values!

-_-_-_-_-_-_-_,------,   
_-_-_-_-_-_-_-|   /\_/\   NYANYANYAN
-_-_-_-_-_-_-~|__( ^ .^) /
_-_-_-_-_-_-_-""  ""   

*//////////////////////////////////////////////////////////////////////////////
#pragma once


#include "Matrix4.h"
#include "Vector3.h"
namespace Core{
	class Camera	{
	public:
		Camera(void){
			yaw = 0.0f;
			pitch = 0.0f;
			//	msec	= 5;
		};

		Camera(float pitch, float yaw, Vector3 position){
			this->pitch = pitch;
			this->yaw = yaw;
			this->position = position;
			//	this->msec		= 5;
		}

		~Camera(void){};

		void UpdateCamera(float msec = 10.0f);

		//Builds a view matrix for the current camera variables, suitable for sending straight
		//to a vertex shader (i.e it's already an 'inverse camera matrix').
		Matrix4 BuildViewMatrix();
		Vector3 BuildOrientation();
		//Gets position in world space
		Vector3 GetPosition() const { return position; }
		//Sets position in world space
		void	SetPosition(Vector3 val) { position = val; }

		//Gets yaw, in degrees
		float	GetYaw()   const { return yaw; }
		//Sets yaw, in degrees
		void	SetYaw(float y) { yaw = y; }

		//Gets pitch, in degrees
		float	GetPitch() const { return pitch; }
		//Sets pitch, in degrees
		void	SetPitch(float p) { pitch = p; }
		void keyboardEvents(char key);
		void mouseMoveEvents(int x, int y);
	protected:
		float	yaw;
		float	pitch;
		Vector3 position;
		//	float msec;

	};
}
