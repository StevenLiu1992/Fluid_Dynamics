//Init_GLUT.h
#pragma once
#include "ContextInfo.h"
#include "FrameBufferInfo.h"
#include "WindowInfo.h"
#include <iostream>
#include "Init_GLEW.h"
#include "IListener.h"
#include "WaterDefine.h"
#include "WaterKernel.cuh"

namespace Core {
	namespace Init{//two namespaces

		class Init_GLUT{

		public:
			static void init(const Core::WindowInfo& window,
				const Core::ContextInfo& context,
				const Core::FramebufferInfo& framebufferInfo);

		public:
			static void run();//called from outside
			static void close();

			void enterFullscreen();
			void exitFullscreen();
	//		static Camera* camera;
			//used to print info about GL
			static void printOpenGLInfo(const Core::WindowInfo& windowInfo,
				const Core::ContextInfo& context);
		private:
			static void idleCallback(void);
			static void displayCallback(void);
			static void reshapeCallback(int width, int height);
			static void closeCallback();
			static void keyboardCallback(unsigned char Key, int x, int y);
			static void keyboardUpCallback(unsigned char key, int x, int y);
			static void mouseMove(int x, int y);
			static void click(int button, int updown, int x, int y);
			static void motion(int x, int y);
		private:
			static Core::IListener* listener;
			static Core::WindowInfo windowInformation;
			static bool wPressed;
			static bool sPressed;
			static bool aPressed;
			static bool dPressed;
			static bool qPressed;
			static bool ePressed;
			static int xOrigin;
			static int yOrigin;
			static int delta_x;
			static int delta_y;
			static bool mouseFirstMotion;
			static int lastx;
			static int lasty;
			static bool clicked;

			static int wWidth;
			static int wHeight;
		public:
			static void setListener(Core::IListener*& iListener);
		};
	}
}