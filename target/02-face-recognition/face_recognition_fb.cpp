#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h> // For memcpy
#include <unistd.h>
#include <sys/select.h>
#include <termios.h>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm> // for std::min

#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp> // From opencv_contrib

// --- [ 關鍵設定 (必須匹配 Python 訓練腳本 trainnnn.py) ] ---

// 偵測器：必須和 Python 訓練時用的 (haarcascade_frontalface_default.xml) 一致
const std::string CASCADE_PATH = "haarcascade_frontalface_default.xml";
// 模型：必須是 Python 訓練輸出的 (trainer.yml)
const std::string MODEL_PATH = "trainer3.yml";

// *** 關鍵閾值 ***
// 填入你昨晚在 Mac 上用 Python (test_model_mac.py) 測試出的最佳數值
// 你的 trainnnn.py 預設是 70.0，請依測試結果調整
const double RECOGNITION_THRESHOLD = 80.0; 

// *** 關鍵尺寸 ***
// 必須和 Python 訓練時的 (width, height) = (200, 200) 一致
const cv::Size TRAINING_IMAGE_SIZE(200, 200);

// --- [ 設備設定 ] ---
const int CAMERA_DEVICE_ID = 2; // 攝影機 ID (例如 /dev/video0)
const char* FB_DEVICE_PATH = "/dev/fb0"; // Framebuffer 設備
const int FRAME_RATE = 10; // 攝影機 FPS

/**
 * @brief 讀取 Framebuffer 設備的螢幕資訊
 * *** [ 修正 1/3 ] ***
 * 新增 line_pitch 以讀取真實的行距
 */
struct framebuffer_info
{
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t yres_virtual;
    uint32_t line_pitch; // 新增：儲存真實的 line_length
};

/**
 * @brief 透過 ioctl 獲取 framebuffer 資訊
 * @param framebuffer_device_path 設備路徑 (例如 "/dev/fb0")
 * @return 包含螢幕資訊的 struct
 *
 * *** [ 修正 2/3 ] ***
 * 修改此函數以讀取 screen_info.line_length
 */
struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path) {
    struct framebuffer_info fb_info;
    struct fb_var_screeninfo screen_info;
    struct fb_fix_screeninfo fix_info; // *** 1. 宣告 fix_info ***
    
    int fbfd = open(framebuffer_device_path, O_RDWR);
    if (fbfd == -1) {
        perror("錯誤: 無法開啟 framebuffer 設備");
        exit(1);  
    }

    if (ioctl(fbfd, FBIOGET_VSCREENINFO, &screen_info) == -1) {
        perror("錯誤: 讀取 VSCREENINFO 失敗");
        close(fbfd);
        exit(1);
    }

    // *** 2. 新增：取得 FSCREENINFO (固定資訊) ***
    if (ioctl(fbfd, FBIOGET_FSCREENINFO, &fix_info) == -1) {
        perror("錯誤: 讀取 FSCREENINFO 失敗");
        close(fbfd);
        exit(1);
    }
    
    close(fbfd);

    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.yres_virtual = screen_info.yres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;
    // *** 3. 修正：從 fix_info 讀取 line_length ***
    fb_info.line_pitch = fix_info.line_length; // 修正：儲存 line_length
    
    std::cout << "--- [ Framebuffer 資訊 ] ---" << std::endl;
    std::cout << "解析度: " << fb_info.xres_virtual << "x" << fb_info.yres_virtual << std::endl;
    std::cout << "色深 (bpp): " << fb_info.bits_per_pixel << std::endl;
    std::cout << "真實行距 (line_pitch): " << fb_info.line_pitch << " bytes" << std::endl;
    std::cout << "--------------------------" << std::endl;
    
    return fb_info;
}

/**
 * @brief 主程式
 * *** [ 修正 3/3 ] ***
 * 1. C++98: map 初始化、for 迴圈
 * 2. Typo: cv2:: -> cv::
 * 3. Render: 使用 line_pitch 並動態轉換色深
 */
int main(int argc, char *argv[]) {
    // --- 1. 初始化 ---
    cv::Mat frame;

    cv::VideoCapture camera(CAMERA_DEVICE_ID); 
    if (!camera.isOpened()) {
        std::cerr << "錯誤: 無法開啟攝影機 " << CAMERA_DEVICE_ID << std::endl;
        return 1;
    }

    // 載入人臉偵測器 (Haar)
    cv::CascadeClassifier face_detector;
    if (!face_detector.load(CASCADE_PATH)) {
        std::cerr << "錯誤: 無法從 " << CASCADE_PATH << " 載入偵測器" << std::endl;
        return -1;
    }

    // +++ 載入人臉辨識模型 (LBPH) +++
    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    
    try {
        model->read(MODEL_PATH);
        std::cout << "LBPH 模型載入成功: " << MODEL_PATH << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "錯誤: 無法從 " << MODEL_PATH << " 載入模型" << std::endl;
        std::cerr << "OpenCV 錯誤訊息: " << e.what() << std::endl;
        return 1;
    }

    // --- [ C++98 修正 (Map 初始化) ] ---
    // 原 C++11 寫法: std::map<int, std::string> label_to_name = { ... };
    std::map<int, std::string> label_to_name;
    label_to_name[312553011] = "312553011";
    label_to_name[312554050] = "312554050";
    // 格式: { 學號_整數, "要顯示的學號_字串" }
    
    // +++ 結束辨識初始化 +++

    // --- Framebuffer 設定 ---
    framebuffer_info fb_info = get_framebuffer_info(FB_DEVICE_PATH);
    std::ofstream ofs(FB_DEVICE_PATH);
    if (!ofs) {
        std::cerr << "錯誤: 無法寫入 Framebuffer: " << FB_DEVICE_PATH << std::endl;
        return 1;
    }
    
    camera.set(cv::CAP_PROP_FPS, FRAME_RATE);
    const int frame_width = camera.get(cv::CAP_PROP_FRAME_WIDTH);
    const int frame_height = camera.get(cv::CAP_PROP_FRAME_HEIGHT);

    // --- 2. 畫面縮放與置中計算 ---
    const int fb_width = fb_info.xres_virtual;
    const int fb_height = fb_info.yres_virtual;
    const int bytes_per_pixel = fb_info.bits_per_pixel / 8; // 根據 bpp 計算

    float width_ratio = (float)fb_width / frame_width;
    float height_ratio = (float)fb_height / frame_height;
    float scale_factor = std::min(width_ratio, height_ratio); // Letterbox 縮放

    const int scaled_width = frame_width * scale_factor;
    const int scaled_height = frame_height * scale_factor;

    const int x_offset = (fb_width - scaled_width) / 2;
    const int y_offset = (fb_height - scaled_height) / 2;
    
    cv::Mat resized_frame;
    
    // --- 效能優化：全螢幕緩衝區 ---
    // [修正] 使用 line_pitch * height 計算緩衝區總大小，確保安全
    const int fb_total_bytes = fb_info.line_pitch * fb_info.yres_virtual;
    std::vector<char> screen_buffer(fb_total_bytes, 0); // 初始化為全黑

    // [修正] 使用從 ioctl 讀取的真實行距
    const int line_pitch = fb_info.line_pitch; 
    // [不變] 影像來源 (source) 的行寬 (bytes)
    const int video_line_bytes = scaled_width * bytes_per_pixel; 

    std::cout << "--- [ E9V3 即時辨識啟動 ] ---" << std::endl;
    std::cout << "使用閾值: " << RECOGNITION_THRESHOLD << std::endl;

    // --- 3. 主迴圈 ---
    while (true) {
        if (!camera.read(frame)) {
            std::cerr << "無法讀取影格!" << std::endl;
            break;
        }

        // --- 4. 影像前處理 (匹配 Python 訓練腳本 trainnnn.py) ---
        cv::Mat frame_gray;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        // (不在這裡做 equalizeHist)

        // --- 5. 偵測人臉 (在「原始」灰階圖上偵測) ---
        std::vector<cv::Rect> faces;
        face_detector.detectMultiScale(
            frame_gray, 
            faces, 
            1.1, 
            6, // 匹配 Python 訓練腳本的 minNeighbors=6
            0, 
            cv::Size(80, 80) // 匹配 Python 訓練腳本的 minSize=(80, 80)
        );

        // --- 6. 人臉辨識 (匹配 trainnnn.py 的 preprocess_face) ---
        
        // --- [ C++98 修正 (for 迴圈) ] ---
        // 原 C++11 寫法: for (const auto& face_rect : faces) {
        for (std::vector<cv::Rect>::const_iterator it = faces.begin(); it != faces.end(); ++it) {
            
            // 透過迭代器取得 'face_rect'
            const cv::Rect& face_rect = *it;

            // a) 裁切臉部 (從「原始灰階圖」裁切)
            cv::Mat face_roi_raw = frame_gray(face_rect);
            
            // b) 預處理 (同 Python 訓練腳本的 preprocess_face)
            cv::Mat face_roi_eq;
            cv::equalizeHist(face_roi_raw, face_roi_eq); // 對裁切後的 ROI 均衡化
            
            cv::Mat processed_roi;
            cv::resize(face_roi_eq, processed_roi, TRAINING_IMAGE_SIZE, 0, 0, cv::INTER_CUBIC); 

            // c) 預測
            int predicted_label = -1;
            double confidence = 0.0;
            model->predict(processed_roi, predicted_label, confidence);

            std::string name = "Unknown";
            cv::Scalar box_color = cv::Scalar(0, 0, 255); // 預設：紅色 (Unknown)

            // d) 應用閾值 (LBPH 距離越低越好)
            // 檢查 predicted_label 是否存在於 Map 中
            if (confidence < RECOGNITION_THRESHOLD && label_to_name.count(predicted_label)) {
                name = label_to_name.at(predicted_label); // 透過學號查表
                box_color = cv::Scalar(0, 255, 0); // 綠色 (Recognized)
            }

            // 在影像上繪製文字和方框
            std::stringstream ss;
            ss << name << " C:" << cv::format("%.0f", confidence);
            std::string box_text = ss.str();

            // --- [ C++ 修正 (Typo) ] ---
            // 原寫法: cv2::rectangle / cv2::putText
            cv::rectangle(frame, face_rect, box_color, 2);
            cv::putText(frame, box_text, cv::Point(face_rect.x, face_rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
        }
        // --- 結束辨識 ---

        // --- 7. 影像準備 (Display) ---
        cv::resize(frame, resized_frame, cv::Size(scaled_width, scaled_height));
        
        // --- [ 修正：動態色彩空間轉換 ] ---
        cv::Mat final_frame_formatted; // 用於存放轉換後格式的 Mat
        
        switch (fb_info.bits_per_pixel) {
            case 16:
                // 16-bit, 假設為 BGR565
                cv::cvtColor(resized_frame, final_frame_formatted, cv::COLOR_BGR2BGR565);
                break;
            case 24:
                // 24-bit, 假設為 BGR888 (OpenCV 預設)
                final_frame_formatted = resized_frame; // 無需轉換
                break;
            case 32:
                // 32-bit, 假設為 BGRA8888 (常見於 HDMI)
                cv::cvtColor(resized_frame, final_frame_formatted, cv::COLOR_BGR2BGRA);
                break;
            default:
                // 不支援的格式
                std::cerr << "警告: 不支援的 bpp: " << fb_info.bits_per_pixel << ". 嘗試 BGR." << std::endl;
                final_frame_formatted = resized_frame;
                break;
        }

        // --- 8. 高效能渲染 (寫入 Framebuffer) ---
        
        // 8.1. 用黑色填滿整個螢幕緩衝區
        std::fill(screen_buffer.begin(), screen_buffer.end(), 0);

        // 8.2. 計算影像在緩衝區中的起始位置
        // [修正] 使用 ioctl 讀取的 line_pitch
        char* buffer_start_pos = screen_buffer.data() + (y_offset * line_pitch) + (x_offset * bytes_per_pixel);

        // 8.3. 逐行複製影像資料到緩衝區
        for (int y = 0; y < scaled_height; y++) {
            
            // 來源：格式轉換後的影像的第 y 行
            const char* source_row_ptr = reinterpret_cast<const char*>(final_frame_formatted.ptr(y));
            
            // 目的：螢幕緩衝區中對應的行 (使用正確的 line_pitch)
            char* dest_row_ptr = buffer_start_pos + (y * line_pitch);

            // 複製整行
            memcpy(dest_row_ptr, source_row_ptr, video_line_bytes);
        }

        // 8.4. 一次性將整個緩衝區寫入 Framebuffer
        ofs.seekp(0); 
        ofs.write(screen_buffer.data(), screen_buffer.size());
    }

    camera.release();
    ofs.close(); // 關閉 framebuffer
    return 0;
}

