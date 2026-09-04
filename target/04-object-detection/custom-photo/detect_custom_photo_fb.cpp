/**
 * Final Lab Part 2: Custom 30-Class YOLOv3 Photo Detection
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm> // Added for std::min
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdio.h>

// --- OpenCV Includes ---
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// --- Configuration ---
// Uncomment to disable framebuffer for testing on PC/QEMU
// #define QEMU_TEST_BUILD

std::string modelWeights = "yolov3_custom_10000_v1.weights";
std::string modelConfig = "yolov3_custom_v1.cfg";
std::string classesFile = "obj.names";

// --- File Configuration ---
std::string inputImageFile = "test.jpg";
// Output file for the Final Lab result
std::string outputImageFile = "result.jpg";

// --- Detection Parameters (UPDATED TO 640x640) ---
float confThreshold = 0.1;
float nmsThreshold = 0.6;
int inpWidth = 640;   // Changed from 416
int inpHeight = 640;  // Changed from 416

std::vector<std::string> classes;

// --- Framebuffer Structs ---
struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t yres_virtual;
};

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

    std::cout << "Framebuffer info: " << fb_info.xres_virtual << "x" << fb_info.yres_virtual
              << ", " << fb_info.bits_per_pixel << " bpp" << std::endl;

    return fb_info;
}

// --- YOLOv3 Helper Functions ---
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net)
{
    static std::vector<cv::String> names;
    if (names.empty())
    {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<cv::String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty() && classId < classes.size())
    {
        label = classes[classId] + ": " + label;
    }
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::putText(frame, label, cv::Point(left, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
}

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// --- Letterbox Helper Function (Fixed for GCC 5.3) ---
cv::Mat letterbox(const cv::Mat& src, int target_width, int target_height) {
    int w = src.cols;
    int h = src.rows;

    // 1. Calculate scale factor
    float scale = std::min((float)target_width / w, (float)target_height / h);

    // FIX: Use standard casting instead of std::round for compatibility
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);

    // 2. Resize
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    // 3. Create a canvas with gray background (128)
    cv::Mat dst(target_height, target_width, CV_8UC3, cv::Scalar(128, 128, 128));

    // 4. Paste resized image into the center
    int dx = (target_width - new_w) / 2;
    int dy = (target_height - new_h) / 2;
    resized.copyTo(dst(cv::Rect(dx, dy, new_w, new_h)));

    return dst;
}


int main(int argc, char** argv)
{
    // --- 1. Load Class Names ---
    std::ifstream ifs(classesFile.c_str());
    if (!ifs.is_open())
    {
        std::cerr << "Error: Could not open classes file: " << classesFile << std::endl;
        return -1;
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    // --- 2. Load the YOLOv3 Network ---
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Network loaded successfully." << std::endl;

    // --- 3. Initialize Framebuffer ---
#ifdef QEMU_TEST_BUILD
    std::cout << "[QEMU_TEST_BUILD] Framebuffer code disabled." << std::endl;
#else
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");
    if (!ofs) {
        std::cerr << "Could not open framebuffer /dev/fb0 for writing." << std::endl;
        return 1;
    }
#endif

    // --- 4. Load Image & Apply Letterbox ---
    cv::Mat raw_frame = cv::imread(inputImageFile);
    if (raw_frame.empty())
    {
        std::cerr << "Error: Could not read image file: " << inputImageFile << std::endl;
        return -1;
    }
    std::cout << "Loaded raw image: " << inputImageFile << " (" << raw_frame.cols << "x" << raw_frame.rows << ")" << std::endl;

    // Resize to 640x640 while maintaining aspect ratio (Letterbox)
    cv::Mat frame = letterbox(raw_frame, inpWidth, inpHeight);
    std::cout << "Letterboxed image created: " << frame.cols << "x" << frame.rows << std::endl;

    cv::Mat blob;

    // --- 5. Run YOLOv3 Inference ---
    // Note: We use the 'frame' (which is now 640x640), so crop=false
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    std::cout << "Starting inference (640x640)... this may take a moment." << std::endl;
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Post-process draws boxes on 'frame' (the letterboxed image)
    postprocess(frame, outs);
    std::cout << "Inference complete." << std::endl;

    // --- 6. Save Final Result to File ---
    bool saveSuccess = cv::imwrite(outputImageFile, frame);
    if (!saveSuccess) {
        std::cerr << "Error: Could not save result to " << outputImageFile << std::endl;
    } else {
        std::cout << "Saved detection results to " << outputImageFile << std::endl;
    }

    // --- 7. Prepare for Framebuffer Display ---
#ifdef QEMU_TEST_BUILD
    const int fb_width = 1024;
    const int fb_height = 768;
    const int bytes_per_pixel = 2;
#else
    const int fb_width = fb_info.xres_virtual;
    const int fb_height = fb_info.yres_virtual;
    const int bytes_per_pixel = fb_info.bits_per_pixel / 8;
#endif

    // Define dimensions of the image we want to display (the letterboxed result)
    const int frame_width = frame.cols;
    const int frame_height = frame.rows;

    // Calculate scale factor to fit the letterboxed image (640x640) onto the screen
    float width_ratio = (float)fb_width / frame_width;
    float height_ratio = (float)fb_height / frame_height;
    float scale_factor = std::min(width_ratio, height_ratio);

    const int scaled_width = (int)(frame_width * scale_factor);
    const int scaled_height = (int)(frame_height * scale_factor);

    const int x_offset = (fb_width - scaled_width) / 2;
    const int y_offset = (fb_height - scaled_height) / 2;

    std::cout << "Display scaling: " << scaled_width << "x" << scaled_height << std::endl;

    // Buffers for writing to framebuffer
    cv::Mat resized_frame, final_frame_bgr565;
    std::vector<char> black_line(fb_width * bytes_per_pixel, 0);
    std::vector<char> black_bar(x_offset * bytes_per_pixel, 0);

    // --- 8. Display Frame on LCD ---
#ifdef QEMU_TEST_BUILD
    std::cout << "[QEMU_TEST_BUILD] Framebuffer display skipped." << std::endl;
#else
    // Resize the letterboxed result to fit the screen
    cv::resize(frame, resized_frame, cv::Size(scaled_width, scaled_height));

    // Convert to BGR565 (16-bit)
    cv::cvtColor(resized_frame, final_frame_bgr565, cv::COLOR_BGR2BGR565);

    // Write to framebuffer
    for (int y = 0; y < fb_height; y++)
    {
        long position = y * fb_width * bytes_per_pixel;
        ofs.seekp(position);

        if (y < y_offset || y >= (y_offset + scaled_height)) {
            // Top/Bottom black bars
            ofs.write(black_line.data(), black_line.size());
        }
        else {
            // Image area
            ofs.write(black_bar.data(), black_bar.size()); // Left padding

            int source_row = y - y_offset;
            ofs.write(reinterpret_cast<const char*>(final_frame_bgr565.ptr(source_row)), scaled_width * bytes_per_pixel);

            ofs.write(black_bar.data(), black_bar.size()); // Right padding
        }
    }
    std::cout << "Result displayed on framebuffer." << std::endl;
#endif

    // --- 9. Wait for Exit ---
    std::cout << "Press ENTER to exit." << std::endl;
    getchar();

#ifndef QEMU_TEST_BUILD
    ofs.close();
#endif

    return 0;
}
