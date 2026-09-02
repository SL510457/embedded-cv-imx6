#include <fcntl.h> 
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <unistd.h>

struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

int main(int argc, const char *argv[]) {
    cv::Mat image;
    cv::Mat processed_image;
    cv::Size2f image_size;
    
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");

    image = cv::imread("sample.bmp", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not read image file." << std::endl;
        return 1;
    }

    image_size = image.size();

    cv::cvtColor(image, processed_image, cv::COLOR_BGR2BGR565);

    for (int y = 0; y < image_size.height; y++)
    {
        long position = y * fb_info.xres_virtual * (fb_info.bits_per_pixel / 8);
        ofs.seekp(position);

        ofs.write(reinterpret_cast<const char*>(processed_image.ptr(y)), image_size.width * (fb_info.bits_per_pixel / 8));
    }

    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path) {
    struct framebuffer_info fb_info;
    struct fb_var_screeninfo screen_info;
    int fd = -1;

    fd = open(framebuffer_device_path, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Error: Could not open framebuffer device: " << framebuffer_device_path << std::endl;
        exit(EXIT_FAILURE);
    }

    if (ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) == -1) {
        std::cerr << "Error: Failed to get screen info using ioctl." << std::endl;
        close(fd);
        exit(EXIT_FAILURE);
    }
    close(fd);

    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;

    return fb_info;
};
