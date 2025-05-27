#include "visage/app.h"
#include "../simulator/simulator.h"
#include "viewer.h"
#include "editor.h"
#include <memory>

class VIRSGUI {
public:
	VIRSGUI();
	~VIRSGUI();
	void showAndRun(int width, int height);
	void renderImage(const Vec3f& cameraPos);

private:
	std::unique_ptr<visage::ApplicationWindow> app;
	std::unique_ptr<Simulator> simulator;
	size_t image_size;
	unsigned char* image;
	std::unique_ptr<visage::Frame> container;
	std::unique_ptr<Viewer> viewer;
	std::unique_ptr<Editor> editor;
	visage::Palette palette;

};