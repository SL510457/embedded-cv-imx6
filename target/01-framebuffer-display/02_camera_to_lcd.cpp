#include <fcntl.h> 
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <sys/select.h>
#include <dirent.h>
#include <unistd.h>

struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

int find_next_dir_num_by_scanning(const char *path_to_scan, const char *basename) {
    DIR *dir;
    struct dirent *entry;
    int max_num = -1;

    dir = opendir(path_to_scan);
    if (dir == NULL) {
        perror("Error opening directory");
        return -1;
    }

    while ((entry = readdir(dir)) != NULL) {
        int current_num;
        char format[256];
        snprintf(format, sizeof(format), "%s_%%d", basename);

        if (sscanf(entry->d_name, format, &current_num) == 1) {
            char full_path[PATH_MAX];
            snprintf(full_path, sizeof(full_path), "%s/%s", path_to_scan, entry->d_name);
            
            struct stat st;
            if (stat(full_path, &st) == 0 && S_ISDIR(st.st_mode)) {
                if (current_num > max_num) {
                    max_num = current_num;
                }
            }
        }
    }

    closedir(dir);

    return max_num + 1;
}

struct termios orig_termios;

void disable_echo() {
    struct termios newt;
    tcgetattr(STDIN_FILENO, &orig_termios);
    newt = orig_termios;
    newt.c_lflag &= !(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
}


void restore_terminal() {
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
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

int main(int argc, const char *argv[]) {
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");

    cv::VideoCapture camera(2);
    if(!camera.isOpened()) {
        std::cerr << "Could not open video device." << std::endl;
        return 1;
    }
    camera.set(cv::CAP_PROP_FRAME_WIDTH, fb_info.xres_virtual);

    char base_name[50] = "screenshot";
    int maxNum = find_next_dir_num_by_scanning("./", base_name);
    char folderPath[50];
    snprintf(folderPath, sizeof(folderPath), "./%s_%d", base_name, maxNum);

    if (mkdir(folderPath, 0775) == -1) {
        perror("Error creating directory");
        return EXIT_FAILURE;
    } else {
        printf("Directory '%s' created successfully.\n", folderPath);
    }

    disable_echo();

    int frameCounter = 0;

    while(true) {
        cv::Mat frame;
        cv::Mat new_frame;
        cv::Size2f frame_size;
    
        bool ret = camera.read(frame);
        if (!ret){
            std::cerr << "Cannot read frame!" << std::endl;
        }

        frame_size = frame.size();
        int fb_width = fb_info.xres_virtual;
        int fb_depth = fb_info.bits_per_pixel;

        cvtColor(frame, new_frame, cv::COLOR_BGR2BGR565);

        std::vector<char> lineBuffer(fb_width * (fb_depth / 8), 0);
        for (int y = 0; y < frame_size.height; y++)
        {
            long position = y * fb_width * (fb_depth / 8);
            ofs.seekp(position);

            memcpy(lineBuffer.data() + (int)(fb_width - frame_size.width) / 2 * (fb_depth / 8), reinterpret_cast<const char*>(new_frame.ptr(y)), frame_size.width * (fb_depth / 8));
            ofs.write(lineBuffer.data(), fb_width * (fb_depth / 8));
        }
        if (kbhit()) {
            printf("FIRST LAYER\n");
            char key = getch();
            if (key == 'c' || key == 'C') {
                char filename[50];
                snprintf(filename, sizeof(filename), "%s/frame_%d.bmp", folderPath, frameCounter);
                frameCounter++;
                cv::imwrite(filename, frame);
                printf("SCREENSHOT %d\n", frameCounter);
            } else if (key == 'q' || key == 'Q') {
                sync();
                restore_terminal();
                return 0;
            }
         }
    }

    restore_terminal();

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
