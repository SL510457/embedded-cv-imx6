# target/

C++ cross-compiled with `arm-linux-gnueabihf-g++` and run on the E9V3 board.

These programs only make sense on the target. They open `/dev/fb0`, query the display
geometry with `ioctl(FBIOGET_VSCREENINFO)`, convert frames to the framebuffer's pixel
format by hand, and write the bytes directly. They do not build on a desktop: `<linux/fb.h>`
is Linux-only, and the OpenCV 2.x constants in `03_camera_to_hdmi.cpp`
(`CV_CAP_PROP_FPS`, `CV_FOURCC`) were removed in OpenCV 4.

Build and deployment commands are in the top-level README.

| | |
|---|---|
| `01-framebuffer-display/` | Lab 2 — framebuffer output, camera capture, HDMI, scrolling display |
| `02-face-recognition/` | Lab 3.1 — real-time LBPH face recognition |
| `03-helmet-detection/` | Lab 3.2 — YOLOv3 helmet counting on a high-resolution photo |
| `04-object-detection/realtime-coco/` | Final project Part 1 — live YOLOv3-Tiny COCO detection |
| `04-object-detection/custom-photo/` | Final project Part 2 — custom 30-class YOLOv3 photo detection |

The two final-project programs and their runtime assets are documented in
[`04-object-detection/README.md`](04-object-detection/README.md). Model weights
remain outside Git because each Darknet checkpoint is roughly 235 MB.
