//GlutInit.cpp
#include "Init_GLUT.h"

using namespace Core::Init;


extern "C" void addForces(float2 *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);
extern float2 *dvfield;

Core::IListener* Init_GLUT::listener = NULL;
Core::WindowInfo Init_GLUT::windowInformation;
bool Init_GLUT::wPressed = false;
bool Init_GLUT::sPressed = false;
bool Init_GLUT::aPressed = false;
bool Init_GLUT::dPressed = false;
bool Init_GLUT::qPressed = false;
bool Init_GLUT::ePressed = false;
int Init_GLUT::xOrigin = 0;
int Init_GLUT::yOrigin = 0;
bool Init_GLUT::mouseFirstMotion = true;
int Init_GLUT::delta_x = 0;
int Init_GLUT::delta_y = 0;

int  Init_GLUT::lastx = 0;
int  Init_GLUT::lasty = 0;
bool Init_GLUT::clicked = false;

int Init_GLUT::wWidth = 0;
int Init_GLUT::wHeight = 0;


void Init_GLUT::init(const Core::WindowInfo& windowInfo,
	const Core::ContextInfo& contextInfo,
	const Core::FramebufferInfo& framebufferInfo)
{
	//we need to create these fake arguments
	int fakeargc = 1;
	char *fakeargv[] = { "fake", NULL };
	glutInit(&fakeargc, fakeargv);

	wWidth = windowInfo.width;
	wHeight = windowInfo.height;

	if (contextInfo.core)
	{
		glutInitContextVersion(contextInfo.major_version,
			contextInfo.minor_version);
		glutInitContextProfile(GLUT_CORE_PROFILE);
	}
	else
	{
		//As I said in part II, version doesn't matter
		// in Compatibility mode
		glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);
	}
	windowInformation = windowInfo;
	//these functions were called in the old main.cpp
	//Now we use info from the structures
	glutInitDisplayMode(framebufferInfo.flags);
	glutInitWindowPosition(windowInfo.position_x,
		windowInfo.position_y);
	glutInitWindowSize(windowInfo.width, windowInfo.height);

	glutCreateWindow(windowInfo.name.c_str());

	std::cout << "GLUT:initialized" << std::endl;
	//these callbacks are used for rendering
	glutIdleFunc(idleCallback);
	glutCloseFunc(closeCallback);
	glutDisplayFunc(displayCallback);
	glutReshapeFunc(reshapeCallback);
	glutKeyboardFunc(keyboardCallback);
	glutKeyboardUpFunc(keyboardUpCallback);
	glutPassiveMotionFunc(mouseMove);
	glutMouseFunc(click);
	glutMotionFunc(motion);

	//init GLEW, this can be called in main.cpp
	Init::Init_GLEW::Init();

	//cleanup
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
		GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	//our method to display some info. Needs contextInfo and windowinfo
	printOpenGLInfo(windowInfo, contextInfo);

}

//starts the rendering Loop
void Init_GLUT::run()
{
	std::cout << "GLUT:\t Start Running " << std::endl;
	glutMainLoop();
}

void Init_GLUT::close()
{
	std::cout << "GLUT:\t Finished" << std::endl;
	glutLeaveMainLoop();
}

void Init_GLUT::click(int button, int updown, int x, int y)
{
	lastx = x;
	lasty = y;
	clicked = !clicked;
}

void Init_GLUT::motion(int x, int y)
{
	// Convert motion coordinates to domain
	float fx = (lastx / (float)wWidth);
	float fy = (lasty / (float)wHeight);
	int nx = (int)(fx * DIM);
	int ny = (int)(fy * DIM);

	if (clicked && nx < DIM - FR && nx > FR - 1 && ny < DIM - FR && ny > FR - 1)
	{
		int ddx = x - lastx;
		int ddy = y - lasty;
		fx = ddx / (float)wWidth;
		fy = ddy / (float)wHeight;
		int spy = ny - FR;
		int spx = nx - FR;

		addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
//		listener->notifyMouseClick(spx, spy, fx, fy);
		lastx = x;
		lasty = y;
	}

	glutPostRedisplay();
}

void Init_GLUT::keyboardCallback(unsigned char key, int x, int y)
{
	
//	camera->keyboardEvents(key);
	switch (key) {
	case 27:{
		//ESC
		glutLeaveMainLoop();
		break;
	}
	case 'w':{	//ESC
		wPressed = true;
		break;
	}
	case 's':{	//ESC
		sPressed = true;
		break;
	}
	case 'a':{	//ESC
		aPressed = true;
		break;
	}
	case 'd':{	//ESC
		dPressed = true;
		break;
	}
	case 'q':{	//ESC
		qPressed = true;
		break;
	}
	case 'e':{	//ESC
		ePressed = true;
		break;
	}
	}
}

void Init_GLUT::keyboardUpCallback(unsigned char key, int x, int y)
{
	
	//	camera->keyboardEvents(key);
	switch (key) {
	case 'w':{	//ESC
		wPressed = false;
		break;
	}
	case 's':{	//ESC
		sPressed = false; 
		break;
	}
	case 'a':{	//ESC
		aPressed = false;
		break;
	}
	case 'd':{	//ESC
		dPressed = false;
		break;
	}
	case 'q':{	//ESC
		qPressed = false;
		break;
	}
	case 'e':{	//ESC
		ePressed = false;
		break;
	}
	}
}

void Init_GLUT::idleCallback(void)
{
	//camera movement
	if (wPressed)
		listener->notifyKeyboardEvent('w');
	if (sPressed)
		listener->notifyKeyboardEvent('s');
	if (aPressed)
		listener->notifyKeyboardEvent('a');
	if (dPressed)
		listener->notifyKeyboardEvent('d');
	if (qPressed)
		listener->notifyKeyboardEvent('q');
	if (ePressed)
		listener->notifyKeyboardEvent('e');


	listener->notifyMouseMoveEvent(delta_x, delta_y);
	//stop move when mouse donot move
	delta_x = 0;
	delta_y = 0;
	
	glutPostRedisplay();
}

void Init_GLUT::displayCallback()
{
	
	//check for NULL
	if (listener)
	{
		listener->notifyBeginFrame();
		listener->notifyDisplayFrame();

		glutSwapBuffers();

		listener->notifyEndFrame();
	}
}

void Init_GLUT::reshapeCallback(int width, int height)
{
	if (windowInformation.isReshapable == true)
	{
		if (listener)
		{
			listener->notifyReshape(width,
				height,
				windowInformation.width,
				windowInformation.height);
		}
		windowInformation.width = width;
		windowInformation.height = height;
	}

}

void Init_GLUT::mouseMove(int x, int y) {

	if (mouseFirstMotion){
		//first motion for mouse
		xOrigin = x;
		yOrigin = y;
		mouseFirstMotion = false;
		
	}
	else{
		delta_x = x - xOrigin;
		delta_y = y - yOrigin;
		xOrigin = x;
		yOrigin = y;
	}
}

void Init_GLUT::closeCallback()
{
	close();
}

//set the listener
void Init_GLUT::setListener(Core::IListener*& iListener)
{
	listener = iListener;
}

void Init_GLUT::enterFullscreen()
{
	glutFullScreen();
}

void Init_GLUT::exitFullscreen()
{
	glutLeaveFullScreen();
}

void Init_GLUT::printOpenGLInfo(const Core::WindowInfo& windowInfo,
	const Core::ContextInfo& contextInfo){

	const unsigned char* renderer = glGetString(GL_RENDERER);
	const unsigned char* vendor = glGetString(GL_VENDOR);
	const unsigned char* version = glGetString(GL_VERSION);

	std::cout << "******************************************************               ************************" << std::endl;
	std::cout << "GLUT:Initialise" << std::endl;
	std::cout << "GLUT:\tVendor : " << vendor << std::endl;
	std::cout << "GLUT:\tRenderer : " << renderer << std::endl;
	std::cout << "GLUT:\tOpenGl version: " << version << std::endl;
}