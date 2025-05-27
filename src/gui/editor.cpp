#include "editor.h"

using namespace visage::dimension;

Editor::Editor() {
	layout().setFlex(true);
	layout().setFlexRows(true);
	layout().setPadding(3_px);
	layout().setPaddingTop(15_px);


	editor_x = std::make_unique<visage::TextEditor>();
	editor_y = std::make_unique<visage::TextEditor>();
	editor_z = std::make_unique<visage::TextEditor>();

	editor_x->setDefaultText("x");
	editor_y->setDefaultText("y");
	editor_z->setDefaultText("z");

	editor_x->setText("7");
	editor_y->setText("7");
	editor_z->setText("7");

	editor_x->layout().setFlexGrow(1.f);
	editor_y->layout().setFlexGrow(1.f);
	editor_z->layout().setFlexGrow(1.f);

	coords_container = std::make_unique<visage::Frame>();
	coords_container->layout().setFlex(true);
	coords_container->layout().setFlexRows(false);
	coords_container->layout().setFlexGap(2_px);
	coords_container->layout().setWidth(100_vw);
	coords_container->layout().setHeight(30_px);
	coords_container->addChild(editor_x.get());
	coords_container->addChild(editor_y.get());
	coords_container->addChild(editor_z.get());

	addChild(coords_container.get());

	render_button = std::make_unique<visage::UiButton>("Render");
	render_button->layout().setMarginTop(5_px);
	render_button->setActionButton(true);
	render_button->layout().setDimensions(100_vw, 30_px);
	render_button->onToggle() = [&](visage::Button* button, bool toggled) {
		notifyRenderButtonClicked();
	};

	addChild(render_button.get());

}

Editor::~Editor() {

}

void Editor::draw(visage::Canvas& canvas) {
	canvas.setColor(0xffeeeeee);
	canvas.fill(0, 0, width(), height());
}


void Editor::notifyRenderButtonClicked() {
	EditorState state;
	state.cameraPos.x = editor_x->text().toFloat();
	state.cameraPos.y = editor_y->text().toFloat();
	state.cameraPos.z = editor_z->text().toFloat();
	render_button_clicked.callback(state);
}