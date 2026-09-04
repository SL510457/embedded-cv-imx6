/**
 * Final Lab Part 1: YOLOv3-Tiny Object Detection (Specific Targets Only)
 * -----------------------------------------------------------------
 * Logic: Multi-Check (Allows "weak" target detections over "strong" non-targets)
 * Compatibility: C++98 (No special compiler flags needed)
 * Hardware: Embedsky E9V3 (Linux Framebuffer)
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cstdio> // Added for sprintf

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Linux Framebuffer Headers
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;

// --- CONFIGURATION ---
const int IN_WIDTH = 320;
const int IN_HEIGHT = 320;
const float CONFIDENCE_THRESHOLD = 0.01;
const float NMS_THRESHOLD = 0.4;
const string FB_PATH = "/dev/fb0";
const int CLASS_OFFSET = 0;

// --- FRAMEBUFFER STRUCT ---
struct Framebuffer {
    int fbfd;
    char* fbp;
    struct fb_var_screeninfo vinfo;
    struct fb_fix_screeninfo finfo;
    long screensize;
    int width, height, bpp, line_length;

    Framebuffer() : fbfd(-1), fbp(NULL) {}

    bool init() {
        fbfd = open(FB_PATH.c_str(), O_RDWR);
        if (fbfd == -1) return false;
        if (ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo)) return false;
        if (ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo)) return false;

        width = vinfo.xres;
        height = vinfo.yres;
        bpp = vinfo.bits_per_pixel;
        line_length = finfo.line_length;
        screensize = vinfo.yres_virtual * finfo.line_length;

        fbp = (char*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fbfd, 0);
        if ((long)fbp == -1) return false;
        return true;
    }

    void cleanup() {
        if (fbp && (long)fbp != -1) munmap(fbp, screensize);
        if (fbfd != -1) close(fbfd);
    }
};

vector<String> getOutputsNames(const Net& net) {
    static vector<String> names;
    if (names.empty()) {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

// Letterbox resizing (Maintains Aspect Ratio)
Mat letterbox(const Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = max(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

int main() {
    Framebuffer fb;
    if (!fb.init()) {
        cerr << "Error: Framebuffer init failed." << endl;
        return -1;
    }

    // 1. Load Classes
    string classesFile = "coco.names";
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) {
        if (!line.empty() && line[line.length()-1] == '\r') line.resize(line.length()-1);
        if (!line.empty()) classes.push_back(line);
    }

    // 2. Identify Target Indices (C++98 COMPATIBLE)
    vector<string> target_names;
    target_names.push_back("book");
    target_names.push_back("bottle");
    target_names.push_back("keyboard");
    target_names.push_back("spoon");
    target_names.push_back("cup");

    vector<int> target_indices;

    cout << "--- Filtering for Targets ---" << endl;
    for (size_t i = 0; i < target_names.size(); ++i) {
        vector<string>::iterator it = find(classes.begin(), classes.end(), target_names[i]);
        if (it != classes.end()) {
            int id = distance(classes.begin(), it);
            target_indices.push_back(id);
            cout << "Found: " << target_names[i] << " (ID: " << id << ")" << endl;
        } else {
            cerr << "Warning: " << target_names[i] << " not found in coco.names!" << endl;
        }
    }
    cout << "-----------------------------" << endl;

    // 3. Load Model
    string modelConfig = "yolov3-tiny.cfg";
    string modelWeights = "yolov3-tiny.weights";
    Net net = readNetFromDarknet(modelConfig, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    VideoCapture cap(2);
    if (!cap.isOpened()) return -1;
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_BUFFERSIZE, 1);

    Mat raw_frame, square_frame, blob;
    vector<Mat> outs;

    // Detection containers
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    vector<int> indices;

    Mat final_render;
    int bytes_per_pixel = fb.bpp / 8;
    double freq = cv::getTickFrequency();

    while (true) {
        int64 t_start = cv::getTickCount();

        cap.grab();
        cap >> raw_frame;
        if (raw_frame.empty()) break;

        square_frame = letterbox(raw_frame);

        blobFromImage(square_frame, blob, 1/255.0, Size(IN_WIDTH, IN_HEIGHT), Scalar(0,0,0), true, false);
        net.setInput(blob);
        net.forward(outs, getOutputsNames(net));

        classIds.clear();
        confidences.clear();
        boxes.clear();

        // --- NEW PARSING LOGIC: Multi-Check (C++98 COMPATIBLE) ---
        for (size_t k = 0; k < outs.size(); ++k) {
            float* data = (float*)outs[k].data;
            for (int j = 0; j < outs[k].rows; ++j, data += outs[k].cols) {

                // C++98 Loop instead of range-based for
                for (size_t m = 0; m < target_indices.size(); ++m) {
                    int target_id = target_indices[m];

                    // In YOLO output, scores start at index 5
                    float confidence = data[5 + target_id];

                    if (confidence > CONFIDENCE_THRESHOLD) {

                        // Coordinate calculation
                        int centerX = (int)(data[0] * square_frame.cols);
                        int centerY = (int)(data[1] * square_frame.rows);
                        int width = (int)(data[2] * square_frame.cols);
                        int height = (int)(data[3] * square_frame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        classIds.push_back(target_id);
                        confidences.push_back(confidence);
                        boxes.push_back(Rect(left, top, width, height));
                    }
                }
            }
        }

        // Apply NMS to remove overlapping boxes
        NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

        // Draw results
        for (size_t k = 0; k < indices.size(); ++k) {
            int idx = indices[k];
            Rect box = boxes[idx];

            rectangle(square_frame, box, Scalar(0, 255, 0), 2);

            int id = classIds[idx];
            string name = classes[id];

            char label[100];
            sprintf(label, "%s %.2f", name.c_str(), confidences[idx]);

            int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(square_frame, Point(box.x, box.y - labelSize.height), Point(box.x + labelSize.width, box.y + baseLine), Scalar(0, 255, 0), FILLED);
            putText(square_frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }

        // --- RENDER TO FRAMEBUFFER ---
        if (fb.bpp == 32) cvtColor(square_frame, final_render, COLOR_BGR2BGRA);
        else if (fb.bpp == 16) cvtColor(square_frame, final_render, COLOR_BGR2BGR565);
        else final_render = square_frame;

        int copy_h = min(final_render.rows, fb.height);
        int copy_w = min(final_render.cols, fb.width);
        int render_pitch = copy_w * bytes_per_pixel;

        for (int y = 0; y < copy_h; y++) {
            const char* src = reinterpret_cast<const char*>(final_render.ptr(y));
            char* dst = fb.fbp + (y * fb.line_length);
            memcpy(dst, src, render_pitch);
        }

        int64 t_end = cv::getTickCount();
        double total_time = ((double)(t_end - t_start) / freq) * 1000.0;

        static int count = 0;
        if (++count % 10 == 0) cout << "FPS: " << (1000.0 / total_time) << endl;
    }

    fb.cleanup();
    return 0;
}
