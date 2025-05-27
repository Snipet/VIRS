#pragma once
#include "visage/ui.h"
#include "visage_widgets/text_editor.h"
#include "visage_widgets/button.h"
#include "../simulator/simulator.h"

struct EditorState {
	Vec3f cameraPos;
};

class Editor : public visage::Frame {
public:
	Editor();
	~Editor();
	void draw(visage::Canvas& canvas) override;
	auto& onRenderButtonClick() { return render_button_clicked; }
	void notifyRenderButtonClicked();

private:
	std::unique_ptr<visage::TextEditor> editor_x;
	std::unique_ptr<visage::TextEditor> editor_y;
	std::unique_ptr<visage::TextEditor> editor_z;
	std::unique_ptr<visage::Frame> coords_container;
	std::unique_ptr<visage::UiButton> render_button;
	visage::CallbackList<void(const EditorState&)> render_button_clicked;
};