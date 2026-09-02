#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath> // Added for std::min

// --- Lab 2 Headers ---
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdio.h> // For getchar()
// ---------------------

// --- NEW ---
// Uncomment this line to build a "test" version for QEMU
// This will disable the framebuffer code (which would crash QEMU)
#define QEMU_TEST_BUILD
// -----------

// --- OpenCV Includes ---
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
// -----------------------

// --- Model Configuration ---
std::string modelWeights = "yolov3/yolov3-obj_2400.weights";
std::string modelConfig = "yolov3/yolov3-obj.cfg";
std::string classesFile = "yolov3/obj.names";

// --- File Configuration ---
// This is the input file it will load
std::string inputImageFile = "test_image.jpg"; 
// This is the output file it will save (as required by Lab 3.2.3)
std::string outputImageFile = "result.jpg";

// --- Detection Parameters ---
float confThreshold = 0.1; // Confidence threshold
float nmsThreshold = 0.5;  // Non-maximum suppression threshold
int inpWidth = 608;  // Width of network's input image
int inpHeight = 608; // Height of network's input image

std::vector<std::string> classes;

// --- Framebuffer Struct (from Lab 2-3) ---
struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t yres_virtual;
};

// --- Framebuffer Function (from Lab 2-3) ---
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
// --- End of YOLOv3 Functions ---


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

    // --- 3. Initialize Framebuffer (from Lab 2-3) ---
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

    // --- 4. Load Local Image (New) ---
    cv::Mat frame = cv::imread(inputImageFile);
    if (frame.empty())
    {
        std::cerr << "Error: Could not read image file: " << inputImageFile << std::endl;
        return -1;
    }
    std::cout << "Loaded image: " << inputImageFile << std::endl;
    cv::Mat blob;

    // --- 5. Run YOLOv3 Inference (New) ---
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));
    
    // Post-process to draw boxes *on the original frame*
    postprocess(frame, outs);
    std::cout << "Inference complete." << std::endl;

    // --- 6. Save Result to File (New, for Lab 3.2.3) ---
    bool saveSuccess = cv::imwrite(outputImageFile, frame);
    if (!saveSuccess) {
        std::cerr << "Error: Could not save result to " << outputImageFile << std::endl;
    } else {
        std::cout << "Saved detection results to " << outputImageFile << std::endl;
    }

    // --- 7. Get Frame & Scaling Info (from Lab 2-3) ---
#ifdef QEMU_TEST_BUILD
    // Set dummy values for test build
    const int fb_width = 1024;
    const int fb_height = 768;
    const int bytes_per_pixel = 2;
#else
    const int fb_width = fb_info.xres_virtual;
    const int fb_height = fb_info.yres_virtual;
    const int bytes_per_pixel = fb_info.bits_per_pixel / 8;
#endif

    // --- FIX ---
    // Define frame_width and frame_height from the loaded image
    const int frame_width = frame.cols;
    const int frame_height = frame.rows;
    // -----------

    // Calculate scale factor to fit video to screen, maintaining aspect ratio
    float width_ratio = (float)fb_width / frame_width;
    float height_ratio = (float)fb_height / frame_height;
    float scale_factor = std::min(width_ratio, height_ratio);

    const int scaled_width = (int)(frame_width * scale_factor);
    const int scaled_height = (int)(frame_height * scale_factor);

    // Calculate offsets for centering
    const int x_offset = (fb_width - scaled_width) / 2;
    const int y_offset = (fb_height - scaled_height) / 2;

    std::cout << "Original image: " << frame_width << "x" << frame_height << std::endl;
    std::cout << "Scaled for display: " << scaled_width << "x" << scaled_height << std::endl;
    std::cout << "Centering offset: (" << x_offset << ", " << y_offset << ")" << std::endl;

    // Buffers for writing to framebuffer
    cv::Mat resized_frame, final_frame_bgr565;
    std::vector<char> black_line(fb_width * bytes_per_pixel, 0);
    std::vector<char> black_bar(x_offset * bytes_per_pixel, 0);

    // --- 8. Display Frame (from Lab 2-3) ---
    
#ifdef QEMU_TEST_BUILD
    std::cout << "[QEMU_TEST_BUILD] Framebuffer display skipped." << std::endl;
#else
    // 1. Resize the *processed* frame to fit the screen
    cv::resize(frame, resized_frame, cv::Size(scaled_width, scaled_height));

    // 2. Convert to BGR565 (16-bit), which your Lab 2 code used
    //    This assumes your /dev/fb0 is 16-bit
    cv::cvtColor(resized_frame, final_frame_bgr565, cv::COLOR_BGR2BGR565);

    // 3. Write to framebuffer, line by line, with centering
    for (int y = 0; y < fb_height; y++) 
    {
        long position = y * fb_width * bytes_per_pixel;
        ofs.seekp(position);

        if (y < y_offset || y >= (y_offset + scaled_height)) {
            // This line is part of the top/bottom black bars
            ofs.write(black_line.data(), black_line.size());
        } 
        else { 
            // This line contains the image
            // Write left black bar
            ofs.write(black_bar.data(), black_bar.size());

            // Write the image data for this row
            int source_row = y - y_offset;
            ofs.write(reinterpret_cast<const char*>(final_frame_bgr565.ptr(source_row)), scaled_width * bytes_per_pixel);

            // Write right black bar
            // (This assumes x_offset is the same for both sides)
            ofs.write(black_bar.data(), black_bar.size());
        }
    }
    std::cout << "Result displayed on framebuffer." << std::endl;
#endif
    
    // --- 9. Wait for Exit (New) ---
    std::cout << "Press ENTER to exit." << std::endl;
    getchar();

#ifndef QEMU_TEST_BUILD
    ofs.close(); // Close the framebuffer stream
#endif
    
    return 0;
}
