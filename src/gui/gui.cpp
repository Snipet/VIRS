#include "gui.h"
#include <iostream>

using namespace visage::dimension;

VIRSGUI::VIRSGUI() {
	image = nullptr;
	app = std::make_unique<visage::ApplicationWindow>();
	app->onDraw() = [&](visage::Canvas& canvas) {
		canvas.setColor(0xffeeeeee);
		canvas.fill(0, 0, app->width(), app->height());
		//canvas.image((const char*)image, image_size, 0.f, 0.f, app->width(), app->height());
	};


	app->setPalette(&palette);
	app->palette()->setColor(visage::theme::ColorId::nameIdMap()["UiActionButtonBackground"], 0xff555555);
	app->palette()->setColor(visage::theme::ColorId::nameIdMap()["UiActionButtonBackgroundHover"], 0xff999999);

	simulator = std::make_unique<Simulator>(600, 600);
	simulator->loadObj("C:\\Users\\seant\\Documents\\Projects\\school\\VIRS\\assets\\room8.obj");
	renderImage({ 7,6,7 });

	container = std::make_unique<visage::Frame>();
	viewer = std::make_unique<Viewer>(image, image_size);
	editor = std::make_unique<Editor>();
	container->onDraw() = [f = container.get()](visage::Canvas& canvas) {
		canvas.setColor(0xff0000ff);
		canvas.fill(0, 0, f->width(), f->height());
	};

	editor->onRenderButtonClick() = [&](const EditorState& state) {
		std::cout << state.cameraPos.x << state.cameraPos.y << state.cameraPos.z << std::endl;
		renderImage(state.cameraPos);
		viewer->setImage(image, image_size);
	};

	container->layout().setDimensions(100_vw, 100_vh);
	container->layout().setFlex(true);
	container->layout().setFlexRows(false);
	app->addChild(container.get());

	viewer->layout().setFlexGrow(1.f);
	editor->layout().setWidth(150_px);
	editor->layout().setHeight(100_vh);
	container->addChild(viewer.get());
	container->addChild(editor.get());


}

void VIRSGUI::showAndRun(int width, int height) {
	app->setTitle("VIRS");
	app->show(width, height);
	app->runEventLoop();
}

VIRSGUI::~VIRSGUI() {

}


void VIRSGUI::renderImage(const Vec3f& pos) {
	if (image) {
		free(image);
	}

	image = nullptr;
	image_size = 0;
	simulator->renderImageToMemory(pos, &image, &image_size);
}