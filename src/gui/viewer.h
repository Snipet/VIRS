#pragma once
#include "visage/ui.h"

class Viewer : public visage::Frame {
public:
	Viewer(unsigned char* img, size_t size);
	~Viewer();
	void draw(visage::Canvas& canvas) override;
	void resized();

	void setImage(unsigned char* img, size_t size) {
		image = img;
		image_size = size;
		redraw();
	}

private:
	unsigned char* image;
	size_t image_size;
	visage::Bounds image_bounds;
};