import cv2
import numpy as np
from PIL import Image
import os
from collections import Counter
from pathlib import Path

# ===[ 可調整參數 ]==============================================================

DATASET_DIR = 'dataset3'               # 訓練用資料夾：放人臉圖片（檔名需含 ID，見 iter_image_paths()）
OUTPUT_FILE = 'trainer3.yml'           # 訓練完成後模型輸出路徑
FACE_SIZE = (200, 200)                # 臉部切出後，統一 resize 的大小（LBPH 對尺寸不敏感，但一致性有助穩定）
HAAR_FACE = "haarcascade_frontalface_default.xml"  # Haar cascade 模型檔路徑（人臉偵測用）
THRESHOLD = 70.0  # 預測距離門檻；LBPH 回傳的是距離/誤差（越小越像）。> THRESHOLD 可視為「未知」


# LBPH 參數說明：
# - radius：LBP 鄰域半徑（1 表示以中心像素為圓心，取半徑=1 的鄰居）
# - neighbors：鄰居點數（8 表示圍繞中心像素取 8 點，比較灰階大小形成 8-bit pattern）
# - grid_x, grid_y：將臉切成多少網格，各區塊分別統計直方圖（區域性特徵）
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

# Haar cascade 偵測器：用來從灰階圖中找出臉部位置（矩形框）
detector = cv2.CascadeClassifier(HAAR_FACE)

# 允許處理的影像副檔名（小寫）
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


# def iter_image_paths(root: str):
#     """
#     目的：遍歷資料夾下的影像檔，並解析檔名中的「人 ID」。

#     支援的檔名格式（扁平式命名）：
#       dataset/<id>.<idx>.jpg  e.g., dataset/313551156.123.jpg
#     - <id>   ：人員識別碼（整數）
#     - <idx>  ：任意序號（不使用，只是避免重名）

#     回傳：
#       逐一 yield (影像完整路徑 Path 物件, person_id:int)

#     設計考量：
#     - 若遇到非影像檔或檔名不符合（第一段非整數），直接跳過。
#     - 使用 Pathlib 使路徑處理更穩健。
#     """
#     rootp = Path(root)
#     for imgp in sorted(rootp.iterdir()):
#         # 僅處理檔案且副檔名符合允許清單
#         if imgp.is_file() and imgp.suffix.lower() in VALID_EXTS:
#             parts = imgp.name.split(".")
#             # 期待格式為 <id>.<idx>.<ext>，因此第一段應可轉成 int
#             try:
#                 person_id = int(parts[0])
#             except Exception:
#                 # 檔名第一段不是整數 ID，略過該檔
#                 continue
#             yield imgp, person_id

def iter_image_paths(root: str):
    """
    目的：遍歷資料夾下的影像檔，並解析「資料夾名稱」作為「人 ID」。

    支援的檔名格式（巢狀式命名）：
      dataset/<id>/<filename>.jpg  e.g., dataset/312554050/frame_00000.jpg
    - <id>   ：人員識別碼（整數），來自資料夾名稱
    - <filename>：任意影像檔

    回傳：
      逐一 yield (影像完整路徑 Path 物件, person_id:int)

    設計考量：
    - 若遇到非影像檔或資料夾名稱非整數，直接跳過。
    - 使用 Pathlib 使路徑處理更穩健。
    """
    rootp = Path(root)

    # 1. 遍歷 root (dataset) 底下的所有項目
    for id_dir_p in sorted(rootp.iterdir()):

        # 2. 確保這個項目是「資料夾」
        if not id_dir_p.is_dir():
            continue  # 略過 dataset 底下的檔案（例如 .DS_Store）

        # 3. 嘗試將資料夾名稱轉為 person_id
        try:
            person_id = int(id_dir_p.name)
        except ValueError:
            # 資料夾名稱不是純數字 (例如 'cache', 'logs')，略過
            print(f"[WARN] 略過非 ID 格式的資料夾: {id_dir_p.name}")
            continue

        # 4. 成功取得 person_id，現在遍歷這個 ID 資料夾中的所有影像
        for imgp in sorted(id_dir_p.iterdir()):

            # 5. 僅處理檔案且副檔名符合允許清單
            if imgp.is_file() and imgp.suffix.lower() in VALID_EXTS:
                # 6. Yield 影像路徑 和 來自「父資料夾」的 ID
                yield imgp, person_id


def preprocess_face(gray_img, rect):
    """
    將偵測到的人臉矩形（rect）從灰階影像切出並做前處理：
    1) 切出 ROI
    2) 直方圖均衡化（equalizeHist）→ 提升對比、減少光照影響
    3) 統一 resize 到 FACE_SIZE
    """
    x, y, w, h = rect
    face = gray_img[y:y + h, x:x + w]                             # 只取臉部區域
    face = cv2.equalizeHist(face)                                 # 對比度拉齊
    face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_CUBIC)  # 統一尺寸
    return face


def get_images_and_labels(path):
    """
    讀取資料夾內所有影像，對每張影像：
      - 轉灰階（用 PIL 開啟再轉 numpy）
      - 使用 Haar 偵測器偵測臉部（取最大的人臉以減少誤框）
      - 前處理（均衡化 + resize）
      - 累積 (face, id) 到訓練資料

    回傳：
      face_samples: List[np.ndarray]    # N 張處理後的臉（皆為同尺寸灰階）
      ids         : List[int]           # 對應每張臉的人 ID

    額外：
      - miss_detect：計數多少張沒有偵測到臉（常見原因：角度、光照、解析度太小）
    """
    face_samples, ids = [], []
    miss_detect = 0

    for img_path, person_id in iter_image_paths(path):
        # 讀圖：PIL 讀檔 + 轉灰階（'L'），再轉為 numpy uint8
        try:
            pil_img = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"[WARN] 讀取失敗，略過 {img_path}: {e}")
            continue
        img = np.array(pil_img, dtype='uint8')

        # 使用 Haar 偵測臉部
        # 參數說明：
        # - scaleFactor：金字塔縮放比例；1.1 表示每次縮小 10%，可抓不同尺度的臉
        # - minNeighbors：相鄰矩形數門檻，越大越嚴格、誤檢較少；但可能漏檢
        # - minSize：忽略比此尺寸更小的目標（可避免雜訊）
        faces = detector.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=6,      # 略嚴格，降低誤檢
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 若未偵測到臉，計數並略過
        if len(faces) == 0:
            miss_detect += 1
            continue

        # 若偵測到多個臉，選擇面積最大者（通常是主體）
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        face = preprocess_face(img, faces[0])  # 前處理（均衡化 + resize）
        face_samples.append(face)
        ids.append(person_id)

    if miss_detect:
        print(f"[INFO] 有 {miss_detect} 張圖沒有偵測到臉（已略過）。")
    return face_samples, ids


def main():
    """
    主流程：
      1) 掃描資料集並預處理臉部
      2) 輸出各 ID 樣本數，協助檢查標註與分佈
      3) 使用 LBPH 訓練模型
      4) 寫出模型檔 OUTPUT_FILE
      5) 隨機抽樣做簡易自我檢查（非嚴謹評估，只當 sanity check）
    """
    print("[INFO] 掃描資料集並預處理臉部...")
    faces, labels = get_images_and_labels(DATASET_DIR)

    # 無資料可訓練 → 直接結束
    if len(faces) == 0:
        print("[ERROR] 沒有取得任何臉。請檢查資料與偵測器。")
        return

    # 統計每個 ID 的樣本數：幫助檢查是否極度不均衡或誤標
    cnt = Counter(labels)
    print("[INFO] 樣本分佈：")
    for k in sorted(cnt.keys()):
        print(f"  ID={k}: {cnt[k]} 張")
    # 只有一個 ID 幾乎無法辨識誰是誰，因為預測只能在「已知」之間挑最近者
    if len(cnt) < 2:
        print("[WARN] 只有一個 ID 的樣本，模型一定會誤配。")

    # 訓練 LBPH
    print("\n[INFO] 訓練 LBPH 模型中...")
    # setThreshold 用於預測階段（predict），不影響 train 過程
    recognizer.setThreshold(THRESHOLD)
    # faces：List[灰階 2D ndarray]；labels：對應的 int ID
    recognizer.train(faces, np.array(labels, dtype=np.int32))

    # 輸出模型檔（之後可用 recognizer.read() 載入）
    recognizer.write(OUTPUT_FILE)
    print(f"[SUCCESS] 訓練完成，模型儲存：{OUTPUT_FILE}")
    print(f"[NOTE] threshold={THRESHOLD}。預測距離大於此值可視為『未知』。")

    # --- 簡易自我檢查（非嚴謹，僅 sanity check） ------------------------------
    # 從訓練資料中抽幾張回插到模型做 predict，看大致是否合理
    # 注意：這不是「測試集」，因此不代表泛化表現；只是快速檢查 pipeline 是否跑通。
    try:
        import random
        idxs = random.sample(range(len(faces)), min(10, len(faces)))  # 最多抽 10 張
        wrong = 0
        for i in idxs:
            label_true = labels[i]
            pred_label, dist = recognizer.predict(faces[i])           # 回傳 (ID, 距離)
            is_unknown = dist > THRESHOLD                             # 門檻外視為未知
            ok = (pred_label == label_true) and (not is_unknown)      # 既要預測正確也不能被判未知
            if not ok:
                wrong += 1
        print(f"[CHECK] 抽樣 {len(idxs)} 張，自我檢查錯 {wrong} 張（僅供參考）")
    except Exception:
        # 若 random 或 predict 出錯，不讓流程失敗（例如極端情況）
        pass


if __name__ == "__main__":
    # 以腳本方式執行時進入主程式
    main()