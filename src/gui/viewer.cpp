#include "viewer.h"

Viewer::Viewer(unsigned char* img, size_t size) {
	image = img;
	image_size = size;
	image_bounds = visage::Bounds(0, 0, 0, 0);
}

Viewer::~Viewer() {

}

void Viewer::draw(visage::Canvas& canvas) {
	canvas.setColor(0xff222222);
	canvas.fill(0, 0, width(), height());
	//canvas.setColor(0xffffff00);
	//canvas.rectangle(image_bounds.x(), image_bounds.y(), image_bounds.width(), image_bounds.height());
	canvas.setColor(0xffffffff);
	if (image) {
		canvas.image((const char*)image, (int)image_size, image_bounds.x(), image_bounds.y(), image_bounds.width(), image_bounds.height());
	}
}

void Viewer::resized() {
	float square_max_len = std::min(width(), height());
	float centerx = width() / 2.f;
	float centery = height() / 2.f;
	image_bounds = visage::Bounds(centerx - square_max_len / 2.f, centery - square_max_len / 2.f, square_max_len, square_max_len);
	redraw();
}