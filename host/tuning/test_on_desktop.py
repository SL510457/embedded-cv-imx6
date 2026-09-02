import cv2
import numpy as np
import time

# --- [ 關鍵設定 (必須 100% 匹配 trainnnn.py) ] ---

# 偵測器：必須和 Python 訓練時用的 (haarcascade_frontalface_default.xml) 一致
CASCADE_PATH = "haarcascade_frontalface_default.xml"
# 模型：必須是 Python 訓練輸出的 (trainer.yml)
MODEL_PATH = "trainer3.yml"

# *** 這是你今晚要找出的關鍵數值 ***
# LBPH 回傳的是「距離」(Distance)，越低越好 (越像)
# 你的 Python 訓練腳本 (trainnnn.py) 預設是 70.0
RECOGNITION_THRESHOLD = 80.0 

# *** 關鍵標籤映射 (匹配 trainnnn.py) ***
# 您的 trainnnn.py 是使用學號 (例如 312553011) 作為標籤 (Label) 進行訓練的。
# 因此，Python 測試腳本中的 Map 也必須使用學號作為「鍵」(Key)。
LABEL_TO_NAME = {
    312553011: "312553011",
    312554050: "312554050" 
    # 格式: { 學號_整數: "要顯示的學號_字串" }
}

# 尺寸：必須和訓練時的 FACE_SIZE (200, 200) 一致
TRAINING_IMAGE_SIZE = (200, 200) 

# Mac 鏡頭 ID：如果 0 不行，請改成 1 或 2
CAMERA_DEVICE_ID = 2
# --- [ 結束設定 ] ---

def main():
    print("--- 啟動 macOS 測試腳本 (方案 A) ---")
    
    # 1. 載入人臉偵測器 (Haar)
    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    if face_detector.empty():
        print(f"錯誤: 無法從 {CASCADE_PATH} 載入偵測器")
        print("請檢查檔案是否存在，且 opencv-contrib-python 已安裝")
        return

    # 2. 載入訓練好的人臉辨識模型 (LBPH)
    try:
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(MODEL_PATH)
        print(f"模型載入成功: {MODEL_PATH}")
    except cv2.error as e:
        print(f"錯誤: 無法從 {MODEL_PATH} 載入模型。")
        print("請檢查檔案是否存在，且 opencv-contrib-python 已安裝")
        print(f"OpenCV 錯誤訊息: {e}")
        return

    # 3. 開啟鏡頭
    camera = cv2.VideoCapture(CAMERA_DEVICE_ID)
    if not camera.isOpened():
        print(f"錯誤: 無法開啟攝影機 {CAMERA_DEVICE_ID}")
        print(">>> 提示：如果 0 不行，請嘗試將 CAMERA_DEVICE_ID 改成 1 或 2 <<<")
        return

    print("\n--- [ 即時測試開始 ] ---")
    print(f"使用閾值 (Threshold): {RECOGNITION_THRESHOLD}")
    print("按 'q' 鍵退出...")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("無法讀取影格!")
            break

        # 4. 影像前處理 (匹配 trainnnn.py 的 get_images_and_labels)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 5. 偵測人臉 (在「原始」灰階圖上偵測)
        #    參數必須匹配 trainnnn.py 的 detector.detectMultiScale
        faces = face_detector.detectMultiScale(
            frame_gray, 
            scaleFactor=1.3, 
            minNeighbors=6,     # 匹配 trainnnn.py
            minSize=(80, 80)    # 匹配 trainnnn.py
        )

        for (x, y, w, h) in faces:
            # 6. 裁切臉部 (從「原始灰階圖」裁切)
            face_roi_raw = frame_gray[y:y+h, x:x+w]
            
            # 7. 預處理 (同 trainnnn.py 的 preprocess_face 函式)
            #   a) 直方圖均衡化 (在 ROI 上)
            face_roi_eq = cv2.equalizeHist(face_roi_raw)
            #   b) 縮放
            resized_roi = cv2.resize(face_roi_eq, TRAINING_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)

            # 8. 預測
            predicted_label, confidence = model.predict(resized_roi)

            name = "Unknown"
            box_color = (0, 0, 255) # 紅色 (Unknown)

            # 9. 判斷 (距離 < 閾值 且 標籤存在於 Map 中)
            if confidence < RECOGNITION_THRESHOLD and predicted_label in LABEL_TO_NAME:
                name = LABEL_TO_NAME[predicted_label]
                box_color = (0, 255, 0) # 綠色 (Recognized)
            
            # 顯示學號和信心指数 (Confidence)
            box_text = f"{name} C: {confidence:.0f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, box_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # 10. 顯示結果
        cv2.imshow("Face Recognition Test (Press 'q' to exit)", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("--- 測試結束 ---")

if __name__ == "__main__":
    main()