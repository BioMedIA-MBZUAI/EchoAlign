# EchoAlign model package

This package loads the EchoAlign weights and encodes echo clips, studies, and text.
For a worked example, see [notebooks/01_quickstart.ipynb](../notebooks/01_quickstart.ipynb).

| File | Contents |
|---|---|
| `loader.py` | Reads one echo clip (DICOM, video, NIfTI, image, or frame folder) as model input |
| `encoder.py` | Clip encoder: one clip to 1 global and 12 anatomical vectors |
| `pooling.py` | Study pooling: a study's clip vectors to 13 study vectors |
| `model.py` | `EchoAlign`: loads `EchoAlign.pt` and encodes clips, studies, and text |
| `pca.py` | Fits the PCA bases on training-split embeddings and writes them into a weights file |
| `device.py` | Turns off MIOpen on AMD GPUs when it cannot create its lock files |

## What the model returns

For each echo study, EchoAlign returns 13 vectors of 512 numbers:

- One global vector for the whole study.
- One vector for each of 12 anatomical structures: left ventricle, right ventricle, left atrium, right atrium, aortic valve, mitral valve, tricuspid valve, pulmonic valve, pericardium, aortic root, aortic arch, and pulmonary artery.

It also encodes text into the same 512-dimensional space, so you can compare a study with a written description of a finding.

For linear probes, the model reduces the 13 vectors to 832 numbers (64 principal components per vector). The principal components were fitted once on the EchoAlign embeddings of the 60,782 studies in the private cohort's training split, using split membership only, not labels, and ship inside the weights file.

Study pooling can take a view and a DICOM type for each clip. The evaluation passes neither, so every clip in every cohort uses DICOM type 0 (unknown). `EchoAlign.pt` has no view embedding.

## Loading echo clips

`load_echo` reads one clip and returns a tensor of shape `[16, 3, 224, 224]`. It accepts
these formats:

| Format | Extensions | Examples |
|---|---|---|
| DICOM | `.dcm` | MIMIC-IV-Echo |
| Video | `.avi`, `.mp4`, and other formats ffmpeg reads | EchoNet-Dynamic, EchoNet-Pediatric, EchoNet-LVH, HMC-QU |
| NIfTI | `.nii`, `.nii.gz` | CAMUS, CardiacNet, SegRWMA |
| Single image | `.png`, `.jpg`, and other image formats | TMED-2 |
| Folder of frames | A folder of images, read in numeric order when every name is a number (`0.png`, `1.png`, ..., `10.png`), otherwise in file-name order | Exported frame sequences |
| Frame array | `.npz` with a `frames` array `[T, H, W]` or `[T, H, W, 3]` | Private cohort (24 fps, never resampled) |

By default, the loader prepares each clip the way EchoAlign was evaluated:

1. It resamples DICOM and video clips to 24 frames per second (nearest frame), as EchoAlign's
   training data were, then reads 16 consecutive frames, starting at the first frame. Shorter clips repeat
   their last frame, as in training, and a single image is repeated.
2. For DICOM files only, it crops each frame to the box around all ultrasound regions declared
   in the file header, then pads it to a square.
3. It resizes each frame to 224 x 224 pixels (frames that are not padded keep no aspect ratio) and normalizes it.

The video encoder reads the 16 frames as one clip, with temporal attention across all 16
frames.

NIfTI frames are transposed into image orientation; SegRWMA and grayscale CardiacNet files,
which are stored upside down, are also rotated by 180 degrees. Color NIfTI files stored as
YCbCr are converted to RGB. Each NIfTI frame is scaled from its own minimum and maximum to 0-255.

### Loader options

Every step is adjustable with an option of `load_echo(path, ...)` (in `loader.py`), which returns `[n_frames, 3, size, size]`:

| Option | Default | What it does |
|---|---|---|
| `n_frames` | `16` | Number of frames to read. `1` reads only the first frame; `None` reads every frame. |
| `stride` | `1` | Step between the selected frames. |
| `target_fps` | `24` | Frame rate that DICOM and video clips are resampled to (nearest frame) before frames are selected. `None` keeps the stored rate. |
| `size` | `224` | Height and width of each output frame, in pixels (bicubic resizing). |
| `mean`, `std` | `0.5` per channel | Normalization of pixel values scaled to [0, 1]. |
| `crop_to_region` | `True` for DICOM, `False` otherwise | Crops each frame to the ultrasound region in the DICOM header. |
| `pad_to_square` | `True` for DICOM, `False` otherwise | Pads each frame to a square. |
| `rotate_180` | Detected from the path | NIfTI only. Rotates SegRWMA and grayscale CardiacNet files, which are stored upside down. |

`read_frames(path, ...)` takes the same options and returns PIL frames before resizing. It keeps the stored frame rate by default (`target_fps=None`). `.npz` frames are never resampled.

Each baseline uses its own preprocessing instead. See
[Preprocessing and zero-shot method](../BASELINES.md#preprocessing-and-zero-shot-method).
