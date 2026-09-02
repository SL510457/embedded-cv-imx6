import cv2
import numpy as np
import sys

# 請確認檔名正確
weights_path = "yolov3_custom_last.weights"
config_path = "yolov3_custom.cfg"

print(f"🔍 正在檢查 {weights_path} 的健康狀況...")

try:
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    last_layer_id = net.getLayerId(layer_names[-1])
    last_layer = net.getParam(last_layer_id)
    
    # 隨機抽查第一層卷積層的權重
    # 注意：YOLOv3 第一層通常叫 'conv_0' 或類似名稱，我們抓取第一層的參數
    model_weights = net.getParam(net.getLayerId(layer_names[0]))
    
    if model_weights is None:
        print("❌ 無法讀取權重參數，可能 OpenCV 版本或讀取方式不支援直接檢查。")
    else:
        # 轉換成 numpy array
        data = np.array(model_weights)
        
        print(f"📊 權重數值範例 (前 10 個): {data.flatten()[:10]}")
        print(f"📈 最大值: {np.max(data)}")
        print(f"📉 最小值: {np.min(data)}")
        print(f"x 平均值: {np.mean(data)}")

        # 關鍵檢查：是否有 NaN (Not a Number)
        if np.isnan(data).any():
            print("\n💀💀💀 診斷結果：死亡 (Contains NaNs) 💀💀💀")
            print("原因：訓練過程中發生『梯度爆炸』，导致權重數值損壞。")
            print("解法：必須降低 learning_rate 並重新訓練。")
        elif np.max(data) == 0 and np.min(data) == 0:
            print("\n👻👻👻 診斷結果：全零 (All Zeros) 👻👻👻")
            print("原因：訓練根本沒寫入任何東西，或者權重讀取錯誤。")
        else:
            print("\n✅✅✅ 診斷結果：數值看起來正常")
            print("如果數值正常但還是測不到，那可能是 .cfg 裡的 anchors 設定錯誤，或圖片 Preprocessing 錯誤。")

except Exception as e:
    print(f"❌ 讀取時發生錯誤: {e}")