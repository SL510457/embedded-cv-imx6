#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>
#include <termios.h>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <opencv2/videoio.hpp>

struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t yres_virtual;
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

int main(int argc, const char *argv[]) {
    const int frame_rate = 10;
    cv::Mat frame;

    cv::VideoCapture camera(2);
    if (!camera.isOpened()) {
        std::cerr << "Could not open video device." << std::endl;
        return 1;
    }

    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");
    if (!ofs) {
        std::cerr << "Could not open framebuffer for writing." << std::endl;
        return 1;
    }
    
    camera.set(CV_CAP_PROP_FPS, frame_rate);

    const int frame_width = camera.get(CV_CAP_PROP_FRAME_WIDTH);
    const int frame_height = camera.get(CV_CAP_PROP_FRAME_HEIGHT);
    
    cv::VideoWriter video("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), frame_rate, cv::Size(frame_width, frame_height), true);
    if (!video.isOpened()) {
        std::cerr << "Could not open VideoWriter." << std::endl;
        return 1;
    }

    const int fb_width = fb_info.xres_virtual;
    const int fb_height = fb_info.yres_virtual;
    const int bytes_per_pixel = fb_info.bits_per_pixel / 8;

    float width_ratio = (float)fb_width / frame_width;
    float height_ratio = (float)fb_height / frame_height;
    float scale_factor = std::min(width_ratio, height_ratio);

    const int scaled_width = frame_width * scale_factor;
    const int scaled_height = frame_height * scale_factor;

    const int x_offset = (fb_width - scaled_width) / 2;
    const int y_offset = (fb_height - scaled_height) / 2;

    std::cout << "Framebuffer: " << fb_width << "x" << fb_height << std::endl;
    std::cout << "Original video: " << frame_width << "x" << frame_height << std::endl;
    std::cout << "Scaled video: " << scaled_width << "x" << scaled_height << std::endl;
    std::cout << "Centering offset: (" << x_offset << ", " << y_offset << ")" << std::endl;
    
    cv::Mat resized_frame, final_frame_bgr565;
    
    while (true) {
        if (!camera.read(frame)) {
            std::cerr << "Cannot read frame!" << std::endl;
            break;
        }

        video.write(frame);

        cv::resize(frame, resized_frame, cv::Size(scaled_width, scaled_height));

        cv::cvtColor(resized_frame, final_frame_bgr565, cv::COLOR_BGR2BGR565);

        for (int y = 0; y < fb_height; y++) {
            long position = y * fb_width * bytes_per_pixel;
            ofs.seekp(position);

            if (y < y_offset || y >= (y_offset + scaled_height)) {
                // Write a full line of black
                std::vector<char> black_line(fb_width * bytes_per_pixel, 0);
                ofs.write(black_line.data(), black_line.size());
            } 
            else { 
                std::vector<char> black_bar(x_offset * bytes_per_pixel, 0);
                ofs.write(black_bar.data(), black_bar.size());

                int source_row = y - y_offset;
                ofs.write(reinterpret_cast<const char*>(final_frame_bgr565.ptr(source_row)), scaled_width * bytes_per_pixel);

                ofs.write(black_bar.data(), black_bar.size());
            }
        }
    }

    camera.release();
    video.release();
    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path) {
    struct framebuffer_info fb_info;
    struct fb_var_screeninfo screen_info;
    
    int fbfd = open(framebuffer_device_path, O_RDWR);
    if (fbfd == -1) {
        perror("Error: cannot open framebuffer device");
        exit(1); 
    }

    if (ioctl(fbfd, FBIOGET_VSCREENINFO, &screen_info) == -1) {
        perror("Error reading variable information");
        close(fbfd);
        exit(1);
    }
    
    close(fbfd);

    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.yres_virtual = screen_info.yres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;
    
    return fb_info;
}
