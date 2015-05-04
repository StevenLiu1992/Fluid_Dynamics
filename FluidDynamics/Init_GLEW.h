//Init_GLEW.h
#pragma once
#include <iostream>
#include "Dependencies/glew/glew.h"
#include "Dependencies/freeglut/freeglut.h"
namespace Core{
	//notice that I added another namespace here(optional)
	namespace Init{

		class Init_GLEW
		{
		public:
			static void Init();
		};
	}
}