//Scene_Manager.cpp
#include "Scene_Manager.h"
using namespace Managers;
#include "Water.h"



Scene_Manager::Scene_Manager()
{
	glEnable(GL_DEPTH_TEST);

	shader_manager = new Shader_Manager();
	shader_manager->CreateProgram("colorShader",
		"../Shaders/Vertex_Shader.glsl",
		"../Shaders/Fragment_Shader.glsl");
	shader_manager->CreateProgram("particleShader",
		"../Shaders/P_Vertex_Shader.glsl",
		"../Shaders/P_Fragment_Shader.glsl");
	shader_manager->CreateProgram("volumeShader",
		"../Shaders/Volume_Vertex_Shader.glsl",
		"../Shaders/Volume_Fragment_Shader.glsl");
	shader_manager->CreateProgram("intersectionShader",
		"../Shaders/Intersection_Vertex_Shader.glsl",
		"../Shaders/Intersection_Fragment_Shader.glsl");

	shader_manager->CreateProgram("velocityFieldShader",
	"../Shaders/VF_Vertex_Shader.glsl",
	"../Shaders/VF_Fragment_Shader.glsl",
	"../Shaders/VF_Geometry_Shader.glsl");
	shader_manager->CreateProgram("DensityShader",
		"../Shaders/VF_Vertex_Shader.glsl",
		"../Shaders/VF_Fragment_Shader.glsl",
		"../Shaders/Billboard_Geometry_Shader.glsl");

	shader_manager->CreateProgram("terrainShader",
		"../Shaders/Terrain_Vertex_Shader.glsl",
		"../Shaders/Terrain_Fragment_Shader.glsl");
	camera = new Core::Camera(0, 0, Vector3(5,10,25));
	models_manager = new Models_Manager(camera);
//	models_manager->setCamera(camera);
	
}

Scene_Manager::~Scene_Manager()
{
	delete shader_manager;
	delete models_manager;
}

//void Scene_Manager::notifyMouseClick(int spx, int spy, float fx, float fy){
//	const Models::Water tempModel = (const Models::Water&)models_manager->GetModel("water");
//	tempModel.addSomeForce(spx, spy, fx, fy);
//	
//}

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