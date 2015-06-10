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

	Models::Water* water = new Models::Water();
	water->SetProgram(Shader_Manager::GetShader("particleShader"));
	water->SetProgram1(Shader_Manager::GetShader("velocityFieldShader"));
	water->SetProgram2(Shader_Manager::GetShader("DensityShader"));
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