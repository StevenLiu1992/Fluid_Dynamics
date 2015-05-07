#pragma once
namespace Core{

	class IListener
	{
	public:
		virtual ~IListener() = 0;

		virtual void notifyKeyboardEvent(unsigned char key) = 0;
		virtual void notifyMouseMoveEvent(int delta_x, int delta_y) = 0;
		virtual void notifyMouseClick(int, int, float, float) = 0;

		virtual void notifyBeginFrame() = 0;
		virtual void notifyDisplayFrame() = 0;
		virtual void notifyEndFrame() = 0;
		virtual void notifyReshape(int width,
			int height,
			int previous_width,
			int previous_height) = 0;
	};

	inline IListener::~IListener(){
		//implementation of pure virtual destructor
	}
}