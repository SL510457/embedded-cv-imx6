# host/

Python run on the development machine, not on the board.

Models were trained and detection parameters tuned on a laptop, then the resulting model
files (`trainer3.yml`, `.weights`) were copied to the SD card and loaded by the C++
programs in `target/`. This split is why `reference/` exists: iterating on confidence
thresholds and NMS parameters takes seconds on a laptop and several minutes per change if
you cross-compile and redeploy each time.

| | |
|---|---|
| `dataset/extract_frames.py` | `.MOV` → individual frames for the face dataset |
| `training/train_lbph.py` | Trains the LBPH model, writes `trainer3.yml` |
| `tuning/test_on_desktop.py` | Webcam harness for choosing the confidence threshold |
| `tuning/inspect_weights.py` | Detects corrupted YOLO checkpoints (NaN / all-zero weights) |
| `reference/detect_helmet.py` | Parameter tuning for helmet detection before the C++ port |
| `reference/detect_photo.py` | Evaluate the final custom 30-class photo model |

The final project also includes two board-side programs under
`target/04-object-detection/`: a live YOLOv3-Tiny COCO detector and a custom
30-class photo detector. Their model selection and artifact hashes are recorded
in [`../docs/model-and-dataset-provenance.md`](../docs/model-and-dataset-provenance.md).

Requires `opencv-contrib-python` — `cv2.face.LBPHFaceRecognizer` is a contrib module.
Note that OpenCV 5.0 removed the Darknet importer these scripts rely on; use 4.x.
