import cv2
import numpy as np
import os

# --- Configuration ---
# Files must be in the same directory as this script
WEIGHTS_PATH = "yolov3_custom_10000_v1.weights"
CONFIG_PATH = "yolov3_custom_v1.cfg"
NAMES_PATH = "obj.names"
# IMAGE_PATHS = ["sample11.jpg", "sample22.png", "1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg", "11.jpg", "12.jpg", "13.jpg", "14.jpg", "15.jpg", "16.jpg", "17.jpg"]
IMAGE_PATHS = ["sample11.jpg", "sample22.png"]

# Thresholds
# Lower confidence allows catching more objects, but increases false positives.
# Since Part 2 allows you to "annotate as many as possible", 0.3 is a good starting point.
CONF_THRESHOLD = 0.0001
NMS_THRESHOLD = 0.4   

def main():
    # 1. Check if files exist
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(CONFIG_PATH):
        print("Error: YOLO files not found! Please download yolov3.weights and yolov3.cfg first.")
        return

    # 2. Load YOLO Model
    print("Loading YOLOv3 model...")
    net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
    
    # Set backend to OpenCV (runs on CPU, compatible with MacBook)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load COCO class names
    with open(NAMES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Generate random colors for each class
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Get output layer names
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        # Compatibility for different OpenCV versions
        output_layers = net.getUnconnectedOutLayersNames()

    # 3. Process each image
    for img_path in IMAGE_PATHS:
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found.")
            continue

        print(f"Processing {img_path}...")
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        # 4. Preprocessing (Blob)
        # 416x416 is standard. For Part 2, you can try 608x608 for better small object detection.
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # 5. Inference (Forward Pass)
        outs = net.forward(output_layers)

        # 6. Post-processing
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > CONF_THRESHOLD:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 7. Non-Maximum Suppression (NMS) to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        # 8. Draw boxes
        detected_items = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence_score = str(round(confidences[i], 2))
                color = colors[class_ids[i]]
                
                # Draw rectangle
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # Draw label background
                cv2.rectangle(img, (x, y - 30), (x + w, y), color, -1)
                # Draw text
                cv2.putText(img, label + " " + confidence_score, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                detected_items.append(label)

        print(f"  -> Found {len(detected_items)} objects: {set(detected_items)}")

        # 9. Save result
        output_filename = "result_" + img_path
        cv2.imwrite(output_filename, img)
        print(f"  -> Saved to {output_filename}\n")

    print("All done! Check the 'result_' images.")

if __name__ == "__main__":
    main()