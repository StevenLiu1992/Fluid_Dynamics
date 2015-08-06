#include "Models_Manager.h"

using namespace Managers;
using namespace Rendering;

Models_Manager::Models_Manager(Core::Camera* c)
{
	camera = c;
	//triangle game object
	Models::Triangle* triangle = new Models::Triangle();
	triangle->SetProgram(Shader_Manager::GetShader("colorShader"));
	triangle->Create();
	gameModelList["triangle"] = triangle;


	Models::Quad* quad = new Models::Quad();
	quad->SetProgram(Shader_Manager::GetShader("colorShader"));
	quad->Create();
	gameModelList["quad"] = quad;

	Models::Cube* cube = new Models::Cube();
	cube->SetProgram(Shader_Manager::GetShader("colorShader"));
	cube->Create();
	gameModelList["cube"] = cube;

	/*Models::Terrain* terrain = new Models::Terrain();
	terrain->SetProgram(Shader_Manager::GetShader("terrainShader"));
	terrain->Create("../Textures/HB_dem_5m.txt", camera);
	gameModelList["terrain"] = terrain;*/

	Models::Water* water = new Models::Water();
	water->SetProgram(Shader_Manager::GetShader("particleShader"));
	water->SetProgram1(Shader_Manager::GetShader("velocityFieldShader"));
	water->SetProgram2(Shader_Manager::GetShader("DensityShader"));
	water->SetColorProgram(Shader_Manager::GetShader("volumeShader"));
	water->SetInterProgram(Shader_Manager::GetShader("intersectionShader"));
	water->Create(camera);
	gameModelList["water"] = water;
	
}

Models_Manager::~Models_Manager()
{
	//auto -it's a map iterator
	for (auto model : gameModelList)
	{
		delete model.second;
	}
	gameModelList.clear();
}

void Models_Manager::DeleteModel(const std::string& gameModelName)
{
	IGameObject* model = gameModelList[gameModelName];
	model->Destroy();
	gameModelList.erase(gameModelName);
}

const IGameObject& Models_Manager::GetModel(const std::string& gameModelName) const
{
	return (*gameModelList.at(gameModelName));
}

void Models_Manager::Update(Matrix4 viewMatrix)
{
	//auto -it's a map iterator
	for (auto model : gameModelList)
	{
		model.second->Update(viewMatrix);
	}
}

void Models_Manager::Draw()
{
	//auto -it's a map iterator
	for (auto model : gameModelList)
	{
		model.second->Draw();
	}
}