#pragma once
#include <map>
#include "Shader_Manager.h"
#include "IGameObject.h"
#include "Triangle.h"
#include "Quad.h"
#include "Water.h"
#include "Terrain.h"
#include "Cube.h"
#include "Matrix4.h"
#include "Vector3.h"
#include "Camera.h"
using namespace Rendering;
namespace Managers
{
	class Models_Manager
	{
	public:
		Models_Manager(Core::Camera* c = NULL);
		~Models_Manager();

		void Draw();
		void Update(Matrix4 viewMatrix);
		void DeleteModel(const std::string& gameModelName);
		const IGameObject& GetModel(const std::string& gameModelName) const;
		void setCamera(Core::Camera* c){camera = c;}
	private:
		std::map<std::string, IGameObject*> gameModelList;
		Core::Camera* camera;
	};
}