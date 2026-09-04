# Model and dataset provenance

This document records what can be established from the retained source files,
model artifacts, Roboflow export metadata, generated outputs, and the two group
reports. It deliberately keeps conflicting historical settings separate.

## Runtime model selection

| Pipeline | Retained evidence | Model used by the final source |
|---|---|---|
| Real-time final project | Final report and `detect_realtime_fb.cpp` | YOLOv3-Tiny, COCO pretrained |
| Custom-photo final project | Target C++, host reference, and generated results | `yolov3_custom_10000_v1.weights` with `yolov3_custom_v1.cfg` |
| Helmet detection | Lab 3 report, Python reference, and target C++ | `yolov3-obj_2400.weights` with `yolov3-obj.cfg` |
| Face recognition | Training and target source | LBPH model serialized as `trainer3.yml` |

The final-project report says YOLOv4-Tiny was attempted for real-time detection
but was not supported adequately by OpenCV 3.4.7 on the target. The submitted
Part 1 source therefore uses YOLOv3-Tiny. The archived host directory contains
full YOLOv3 and YOLOv4 experiments, but those files are not evidence that either
full model was the final real-time target model.

## Archived artifact hashes

Large generated artifacts are not stored in Git. These SHA-256 values identify
the retained local copies:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `yolov3_custom_10000_v1.weights` | 246,930,048 | `ee56e09dd776a67375219fc869582750b898054194dd8d0c6dd32670bdb5f12a` |
| `yolov3_custom_last_v1.weights` | 246,930,048 | `ee56e09dd776a67375219fc869582750b898054194dd8d0c6dd32670bdb5f12a` |
| `yolov3_custom_best_v1.weights` | 246,930,048 | `2610c0a213d74e2cecb0ff0870cae66f4e9e3d7c4e9ee70e1c9c80dd4f3ba663` |
| `yolov3-obj_2400.weights` | 246,305,384 | `10c134a678202bc19d17994bf59feff12a331e27c220686ab5d398d5eba24a38` |

The identical hashes show that the named 10,000-iteration custom checkpoint and
the retained `last` checkpoint are the same bytes. The `best` checkpoint is a
different model. The exact YOLOv3-Tiny files deployed for Part 1 and the final
`trainer3.yml` LBPH artifact are not present in the local archive.

## Custom dataset versions

Several Roboflow exports remain in the original course directory:

| Export | Images | Resize | Notable augmentation |
|---|---:|---|---|
| v1 | 51 | 608 x 608 stretch | None |
| v2 | 3,638 | 416 x 416 stretch | Horizontal flip, +/-15-degree rotation, brightness, Gaussian blur, salt-and-pepper noise |
| v3 | 3,638 | 416 x 416 fit with black edges | v2 transforms plus 90-degree rotations and shear |
| v4 | 3,699 | 416 x 416 fit with black edges | Same recorded transforms as v3 |

The extracted v2 export contains 3,399 training, 159 validation, and 80 test
images. Those counts and transforms are the source for the dataset description
in the repository README. The v4 archive was exported after the retained final
checkpoint and result images were created, so it should not be described as the
training set for that checkpoint.

The retained files do not establish conclusively whether v2 or v3 produced the
final custom weights. Accordingly, the repository describes the verified
3,638-image dataset but does not assign a Roboflow version to the checkpoint.

## Resolution records

Three different resolutions appear in the surviving records:

- The final report describes increasing custom-model training from 416 x 416 to
  832 x 832.
- The retained `yolov3_custom_v1.cfg` declares 416 x 416.
- The submitted Part 2 C++ creates a 640 x 640 inference blob.
- The final host evaluation script uses 416 x 416.

Without the original training command or Colab notebook, none of these should
be rewritten as a single definitive end-to-end resolution. They refer to
different stages or historical versions, and the 832 x 832 report statement
cannot be verified from the retained cfg.

## Repository policy

The repository keeps source code, small network definitions, class metadata,
selected result images, and documentation. Model weights, LBPH models, raw
recordings, datasets, virtual environments, cross-toolchains, system images,
compiled binaries, and third-party source archives remain outside Git.

The custom dataset's archived Roboflow metadata identifies it as CC BY 4.0 and
links to <https://universe.roboflow.com/embeded-system-final/lab_final>.
