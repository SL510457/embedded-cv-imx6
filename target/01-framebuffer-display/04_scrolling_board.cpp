#include <fcntl.h> 
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // framebuffer depth
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
    uint32_t yres_virtual;
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

struct termios orig_termios;

void disable_echo() {
    struct termios newt;
    tcgetattr(STDIN_FILENO, &orig_termios);
    newt = orig_termios;
    newt.c_lflag &= !(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
}

int kbhit(void) {
    struct timeval tv = {0L, 0L};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(0, &fds);
    return select(1, &fds, NULL, NULL, &tv);
}

int getch(void) {
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}

void restore_terminal() {
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
}

int main(int argc, const char *argv[])
{
    cv::Mat image;
    cv::Mat resized_image, final_image;
    cv::Size2f image_size;
    
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");

    // read image file (sample.bmp) from opencv libs.
    // https://docs.opencv.org/3.4.7/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
    // image = .......
    image = cv::imread("advance.png");
    if (image.empty()) {
        std::cerr << "Error: Could not read image file." << std::endl;
        return 1;
    }

    // get image size of the image.
    // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a146f8e8dda07d1365a575ab83d9828d1
    // image_size = ......
    image_size = image.size();

    const int fb_width = fb_info.xres_virtual;
    const int fb_height = fb_info.yres_virtual;
    const int bytes_per_pixel = fb_info.bits_per_pixel / 8;

    float width_ratio = (float)fb_width / image_size.width;
    float height_ratio = (float)fb_height / image_size.height;
    float scale_factor = std::min(width_ratio, height_ratio);

    const int scaled_width = image_size.width * scale_factor;
    const int scaled_height = image_size.height * scale_factor;

    cv::resize(image, resized_image, cv::Size(scaled_width, scaled_height));

    // transfer color space from BGR to BGR565 (16-bit image) to fit the requirement of the LCD
    // https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    // https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0
    cv::cvtColor(resized_image, final_image, cv::COLOR_BGR2BGR565);

    disable_echo();

    int image_start_x = 0;
    const int left_speed = -20;
    const int right_speed = 20;
    int speed = right_speed;

    while(true) {
        for (int y = 0; y < fb_height; y++)
        {
            std::vector<char> lineBuffer(fb_width * bytes_per_pixel, 0);

            const uint16_t* src_pixel_ptr = final_image.ptr<uint16_t>(y);
            uint16_t* dest_pixel_ptr = reinterpret_cast<uint16_t*>(lineBuffer.data());

            for (int fb_x = 0; fb_x < fb_width; fb_x++) {
                int source_x = (image_start_x + fb_x) % scaled_width;
                dest_pixel_ptr[fb_x] = src_pixel_ptr[source_x];
            }
    	    long position = y * fb_width * bytes_per_pixel;
    	    ofs.seekp(position);
            ofs.write(lineBuffer.data(), lineBuffer.size());
        }
        if (kbhit()) {
            char key = getch();
            if (key == 'j' || key == 'J') {
                speed = left_speed;
            } else if (key == 'l' || key == 'L') {
                speed = right_speed;
            } else if (key == 'q' || key == 'Q') {
                return 0;
            }
        }
        image_start_x += speed;
        image_start_x = (image_start_x % scaled_width) + scaled_width;
    }

    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path)
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.
    int fd = -1;

    // open device with linux system call "open()"
    // https://man7.org/linux/man-pages/man2/open.2.html
    fd = open(framebuffer_device_path, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Error: Could not open framebuffer device: " << framebuffer_device_path << std::endl;
        exit(EXIT_FAILURE);
    }

    // get attributes of the framebuffer device thorugh linux system call "ioctl()".
    // the command you would need is "FBIOGET_VSCREENINFO"
    // https://man7.org/linux/man-pages/man2/ioctl.2.html
    // https://www.kernel.org/doc/Documentation/fb/api.txt
    if (ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) == -1) {
        std::cerr << "Error: Failed to get screen info using ioctl." << std::endl;
        close(fd);
        exit(EXIT_FAILURE);
    }
    close(fd);

    // put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    // fb_info.xres_virtual =       // 8
    // fb_info.bits_per_pixel =     // 16
    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.yres_virtual = screen_info.yres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;

    return fb_info;
};
