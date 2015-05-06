//Scene_Manager.cpp
#include "Scene_Manager.h"
using namespace Managers;



Scene_Manager::Scene_Manager()
{
	glEnable(GL_DEPTH_TEST);

	shader_manager = new Shader_Manager();
	shader_manager->CreateProgram("colorShader",
		"../Shaders/Vertex_Shader.glsl",
		"../Shaders/Fragment_Shader.glsl");
	models_manager = new Models_Manager();
	camera = new Core::Camera(0, 0, Vector3(0,0,-1));
}

Scene_Manager::~Scene_Manager()
{
	delete shader_manager;
	delete models_manager;
}

void Scene_Manager::notifyBeginFrame()
{
	
	models_manager->Update(camera->BuildViewMatrix());
}

void Scene_Manager::notifyDisplayFrame()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	models_manager->Draw();
}

void Scene_Manager::notifyMouseMoveEvent(int delta_x, int delta_y){
	camera->mouseMoveEvents(delta_x, delta_y);
}

void Scene_Manager::notifyKeyboardEvent(unsigned char key){
	camera->keyboardEvents(key);
}

void Scene_Manager::notifyEndFrame()
{
	//nothing here for the moment
}

void Scene_Manager::notifyReshape(int width,
	int height,
	int previous_width,
	int previous_height)
{
	//nothing here for the moment 

}