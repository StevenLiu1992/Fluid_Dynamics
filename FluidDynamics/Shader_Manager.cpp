#include "Shader_Manager.h"

using namespace Managers;

std::map<std::string, GLuint> Shader_Manager::programs;

Shader_Manager::Shader_Manager(void){}

Shader_Manager::~Shader_Manager(void){
	std::map<std::string, GLuint>::iterator i;
	for (i = programs.begin(); i != programs.end(); ++i)
	{
		GLuint pr = i->second;
		glDeleteProgram(pr);
	}
	programs.clear();

}

std::string Shader_Manager::ReadShader(const std::string& filename)
{

	std::string shaderCode;
	std::ifstream file(filename, std::ios::in);

	if (!file.good())
	{
		std::cout << "Can't read file " << filename.c_str() << std::endl;
		std::terminate();
	}

	file.seekg(0, std::ios::end);
	shaderCode.resize((unsigned int)file.tellg());
	file.seekg(0, std::ios::beg);
	file.read(&shaderCode[0], shaderCode.size());
	file.close();
	return shaderCode;
}

GLuint Shader_Manager::CreateShader(GLenum shaderType,
	const std::string& source,
	const std::string& shaderName)
{

	int compile_result = 0;

	GLuint shader = glCreateShader(shaderType);
	//check for shader create
	if (shader == 0) {
		fprintf(stderr, "Error creating shader type %d\n", shaderType);
		return 0;
	}
	const char *shader_code_ptr = source.c_str();
	const int shader_code_size = source.size();

	glShaderSource(shader, 1, &shader_code_ptr, &shader_code_size);
	glCompileShader(shader);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_result);

	//check for errors
	if (compile_result == GL_FALSE)
	{

		int info_log_length = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
		std::vector<char> shader_log(info_log_length);
		glGetShaderInfoLog(shader, info_log_length, NULL, &shader_log[0]);
		std::cout << "ERROR compiling shader: " << shaderName.c_str() << std::endl << &shader_log[0] << std::endl;
		return 0;
	}
	return shader;
}

void Shader_Manager::CreateProgram(const std::string& shaderName,
	const std::string& vertexShaderFilename,
	const std::string& fragmentShaderFilename,
	const std::string& geometryShaderFilename)
{

	//read the shader files and save the code
	std::string vertex_shader_code = ReadShader(vertexShaderFilename);
	std::string fragment_shader_code = ReadShader(fragmentShaderFilename);
	std::string geometry_shader_code = "";
	if (!geometryShaderFilename.empty()){
		geometry_shader_code = ReadShader(geometryShaderFilename);
	}
	
	GLuint vertex_shader = CreateShader(GL_VERTEX_SHADER, vertex_shader_code, "vertex shader");
	
	GLuint fragment_shader = CreateShader(GL_FRAGMENT_SHADER, fragment_shader_code, "fragment shader");
	GLuint geometry_shader;
	if (!geometry_shader_code.empty())
		geometry_shader = CreateShader(GL_GEOMETRY_SHADER, geometry_shader_code, "geometry shader");
	int link_result = 0;
	//create the program handle, attatch the shaders and link it
	GLuint program = glCreateProgram();
	//check for create program
	if (program == 0) {
		fprintf(stderr, "Error creating shader program\n");
		return;
	}
	if (!geometry_shader_code.empty())
		glAttachShader(program, geometry_shader);
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);

	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_result);
	//check for link errors
	if (link_result == GL_FALSE)
	{

		int info_log_length = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);
		std::vector<char> program_log(info_log_length);
		glGetProgramInfoLog(program, info_log_length, NULL, &program_log[0]);
		std::cout << "Shader Loader : LINK ERROR" << std::endl << &program_log[0] << std::endl;
		return;
	}
	programs[shaderName] = program;
}

const GLuint Shader_Manager::GetShader(const std::string& shaderName)
{
	//make sure that you check if program exist first
	//before you return it
	return programs.at(shaderName);

}