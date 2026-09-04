# Final project: object detection

The final project had two separate target-side pipelines. Both use OpenCV's
Darknet importer on the i.MX6Q CPU and render their results directly to the
Linux framebuffer.

## Part 1: real-time COCO detection

`realtime-coco/detect_realtime_fb.cpp` captures `/dev/video2`, runs a
YOLOv3-Tiny model, keeps detections for five selected COCO categories (`book`,
`bottle`, `keyboard`, `spoon`, and `cup`), and writes the annotated frame to
`/dev/fb0`.

The final report records that YOLOv4-Tiny was attempted first but was not
compatible with the target's OpenCV 3.4.7 DNN implementation. YOLOv3-Tiny was
the final target model.

Run the executable from a directory containing:

```text
yolov3-tiny.cfg
yolov3-tiny.weights
coco.names
```

Only `coco.names` is kept in Git. The archived course directory no longer
contains the exact YOLOv3-Tiny cfg or weights used on the board, so the full
YOLOv3 files in that archive must not be substituted silently.

## Part 2: custom photo detection

`custom-photo/detect_custom_photo_fb.cpp` loads the custom 30-class YOLOv3
checkpoint, processes one local image, saves `result.jpg`, and displays the
annotated image on the LCD framebuffer.

Run the executable from a directory containing:

```text
yolov3_custom_v1.cfg
yolov3_custom_10000_v1.weights
obj.names
test.jpg
```

The generated weights are intentionally excluded from Git. The retained final
inference code names `yolov3_custom_10000_v1.weights`; that file is byte-for-byte
identical to the archived `yolov3_custom_last_v1.weights` checkpoint. See
[`../../docs/model-and-dataset-provenance.md`](../../docs/model-and-dataset-provenance.md)
for hashes and dataset-version notes.

## Retained implementation constraints

These files preserve the submitted implementation rather than presenting a
newly rewritten detector:

- Part 1 uses a 320 x 320 inference input and memory-mapped framebuffer output.
- Part 2 uses a 640 x 640 inference blob even though the retained cfg declares
  416 x 416. OpenCV can accept a different inference size for this fully
  convolutional network, but 640 x 640 is substantially more expensive on the
  Cortex-A9.
- Part 2 converts output to BGR565 and is therefore the 16-bit LCD path. It
  calculates row offsets from width and bytes per pixel rather than
  `finfo.line_length`; the corrected stride-aware renderer is demonstrated by
  `../02-face-recognition/face_recognition_fb.cpp`.

These constraints should be considered before reusing the code on a different
framebuffer configuration.
