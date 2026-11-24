# Baseline models

This repository evaluates EchoAlign against five published echocardiography foundation models:
**EchoPrime**, **PanEcho**, **EchoCLIP**, **EchoJEPA**, and **EchoFM**.

The official code of each baseline is included as a Git submodule under `models/`:

- `models/EchoPrime` ([echonet/EchoPrime](https://github.com/echonet/EchoPrime))
- `models/PanEcho` ([CarDS-Yale/PanEcho](https://github.com/CarDS-Yale/PanEcho))
- `models/echo_CLIP` ([echonet/echo_CLIP](https://github.com/echonet/echo_CLIP))
- `models/EchoJEPA` ([bowang-lab/EchoJEPA](https://github.com/bowang-lab/EchoJEPA))
- `models/EchoFM` ([SekeunKim/EchoFM](https://github.com/SekeunKim/EchoFM))

Before you set up a baseline, complete the [installation](README.md#installation). Each section below is independent, so you can set up only the models you need. Run every command from the repository root.

---

## 1. EchoPrime

EchoPrime pools video embeddings by view and scores findings by retrieving similar reports from a candidate bank.

EchoPrime embeds videos only and skips still images, as its own pipeline does. TMED-2 has only still images, so for TMED-2 each image is repeated as a clip (`echoprime_stills: repeat` in `configs/cohorts.yaml`). Tables mark these EchoPrime results with a dagger.

```bash
git submodule update --init models/EchoPrime
cd models/EchoPrime
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/candidate_embeddings_p1.pt
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/candidate_embeddings_p2.pt
unzip model_data.zip
mv candidate_embeddings_p1.pt candidate_embeddings_p2.pt model_data/candidates_data/
cd ../..
```

## 2. PanEcho

PanEcho is a video model with supervised task heads for common echocardiographic findings. Its weights download from the v1.0 release through `torch.hub` on first use and are cached. On every load, PanEcho reads its task list (`tasks.pkl`) from its GitHub repository.

```bash
git submodule update --init models/PanEcho
```

## 3. EchoCLIP

EchoCLIP is a vision-language model trained on echo frames and reports. Its weights download from the Hugging Face Hub (`mkaichristensen/echo-clip`) on first use. Its zero-shot prompts are in `models/echo_CLIP/prompts_used.json`. Embedding runs EchoCLIP in 32-bit floats; segmentation features use bfloat16 on GPU.

```bash
git submodule update --init models/echo_CLIP
```

## 4. EchoJEPA

EchoJEPA is a video encoder pretrained on echo videos with latent prediction (V-JEPA 2). It has no text encoder, so it is evaluated with linear probes only. The checkpoint is `vitl-vmix22m-pt220-c55.pt` (ViT-L, 5.1 GB) from the [EchoJEPA checkpoint folder](https://drive.google.com/drive/folders/1RFEXMe8TTcABMBz4H_qtiLB43K9jD_lf).

```bash
git submodule update --init models/EchoJEPA
mkdir -p weights
curl -L -o weights/vitl-vmix22m-pt220-c55.pt \
    "https://drive.usercontent.google.com/download?id=1T_ubAMpDMEByH7V6TJu9iT5Yf9IxlDBB&export=download&confirm=t"
```

## 5. EchoFM

EchoFM is a video encoder pretrained on echo videos with masked autoencoding. It has no text encoder, so it is evaluated with linear probes only. Its model and code are licensed CC BY-NC-ND 4.0. The checkpoint is `echofm_vitl.pth` (ViT-L, 1.4 GB) from [sekeun/EchoFM](https://huggingface.co/sekeun/EchoFM).

```bash
git submodule update --init models/EchoFM
mkdir -p weights
curl -L -o weights/echofm_vitl.pth https://huggingface.co/sekeun/EchoFM/resolve/main/echofm_vitl.pth
```

## Model size

The following table gives the parameters of each baseline, counted on its released weights as `eval/embed/` loads them:

| Model | Video or image encoder | Study step | Text encoder | Total |
|---|---:|---:|---:|---:|
| EchoPrime | 34.6 M | View classifier: 87.6 M | 109.3 M | 231.5 M |
| PanEcho | 42.0 M | Task heads: 0.04 M | - | 42.0 M |
| EchoCLIP | 88.1 M | - | 63.4 M | 151.5 M |
| EchoJEPA | 303.9 M | - | - | 303.9 M |
| EchoFM | 305.7 M | - | - | 305.7 M |

## Preprocessing and zero-shot method

Each model is embedded with its own preprocessing and scored zero-shot with its own method:

| Model | Preprocessing | Zero-shot method |
|---|---|---|
| EchoAlign | 16 consecutive frames at 24 fps, 224 x 224 pixels (see [Loading echo clips](echoalign/README.md#loading-echo-clips)) | Mean cosine similarity with the positive prompts minus the mean with the negative prompts. The study vector used is the output for the finding's heart structure, set per task in `configs/zeroshot_tasks.yaml`, or the global output when the task has no structure |
| EchoPrime | Every second frame of the first 32 (16 frames). Each frame is center-cropped, trimmed by 10% on each side, and resized to 224 x 224 pixels (`crop_and_scale`), then normalized with EchoPrime's mean and standard deviation. DICOM frames are first masked outside the ultrasound region | Report retrieval: EchoPrime finds the 50 reports in its candidate bank most similar to the study's embedding for the finding's report section, and averages their labels |
| PanEcho | 16 consecutive frames from the first frame, BGR channel order, 224 x 224 pixels, ImageNet normalization | PanEcho's supervised output (task head) for the finding, averaged over the study's clips |
| EchoCLIP | Every second frame of the first 40 (up to 20 frames). Each frame is cropped and resized as for EchoPrime (`crop_and_scale`), prepared with EchoCLIP's image transform (`preprocess_val`), and embedded separately | EchoCLIP's released prompts and scoring functions, averaged over the study's clips |
| EchoJEPA | 16 consecutive frames at 8 fps. DICOM frames are first cropped to the ultrasound regions in the header and padded to a square, as for EchoAlign. Each frame is resized so its short side is 256 pixels, center-cropped to 224 x 224 pixels, and given ImageNet normalization | None (linear probing only) |
| EchoFM | 32 consecutive frames, 224 x 224 pixels (DICOM frames are first cropped and padded as for EchoAlign), pixel values scaled to [0, 1] | None (linear probing only) |

EchoPrime, PanEcho, EchoCLIP, and EchoFM read frames at the stored frame rate. Private-cohort `.npz` clips are stored at 24 fps and are not resampled: EchoAlign reads them as stored and EchoJEPA reads every third frame. A file with no stored frame rate is read frame by frame.

EchoJEPA's own evaluations use two 16-frame segments per video and an attentive probe (`models/EchoJEPA/configs/eval/`). Here EchoJEPA gets one 16-frame clip per file, the mean of its output tokens, and the same linear probe as every other model.

For tasks that their own zero-shot method does not cover, the text-based baselines (EchoPrime and EchoCLIP) use EchoAlign's prompts (`configs/echoalign_prompts.yaml`) with their own text encoder, scored with the same rule as EchoAlign (mean cosine similarity with the positive prompts minus the mean with the negative prompts). EchoPrime scores the study vector of the finding's report section (`configs/echoprime_sections.yaml`), or the mean of its section vectors when no section fits. EchoCLIP scores each frame, then averages over frames and over the study's clips.

EchoAlign's prompts are in `configs/echoalign_prompts.yaml`. `scripts/select_prompts.py` chose
up to 3 short sentences per class for each private finding on the private VAL split, from the
hand-written candidates in `configs/prompt_candidates.yaml`. MIMIC-IV-Echo and CardioBench tasks
that repeat a private finding use its prompts; CardioBench tasks with no private match use
CardioBench's own prompts.

The linear probes follow the same procedure for every model. They standardize the features on
`TRAIN`, select the regularization on `VAL` (C for classification by AUROC, C for severity grades
by quadratic weighted kappa, alpha for regression by root-mean-square error), refit on `TRAIN` and
`VAL` together, and score `TEST` once. When `VAL` has a single class, or fewer than two grades, the
default (C = 1) is used. EchoAlign's study features are its 832-number representation. EchoPrime
uses its own study pooling: the view-weighted embedding of the report section that matches each
task (`configs/echoprime_sections.yaml`), or the mean of its 15 section embeddings for tasks
without a section. The other models use the mean of the study's clip embeddings.

Segmentation uses the final feature map of EchoAlign, EchoPrime, PanEcho, and EchoCLIP (EchoJEPA and
EchoFM are not segmented). A linear head and a small shared decoder are trained on CAMUS (LV cavity, LV
myocardium, LA), EchoNet-Dynamic, and EchoNet-Pediatric (LV cavity), at the end-diastolic and end-systolic
frames, with decoder seeds 0, 1, and 2; the learning rate and epoch are chosen on `VAL`. A transfer task
applies a head trained on CAMUS to the two EchoNet datasets. Each video model reads 16 frames around the
annotated frame at the frame rate and stride of its embedding path: EchoAlign at 24 fps with stride 1
(CAMUS clips carry no frame rate and are read at their own rate), EchoPrime at the native rate with
stride 2. PanEcho and EchoCLIP read the annotated frame only.

## Test your own model

Any model that turns one echo file into a fixed-length feature vector can be compared with EchoAlign and the baselines by linear probing. The evaluation needs the following from your model:

| Method | What it does | Required |
|---|---|---|
| `__init__(self, device=None)` | Loads your model and weights on `device` | Yes |
| `preprocess(self, path)` | Reads and prepares one clip | Yes |
| `encode(self, batch)` | Encodes a batch of clips | Yes |
| `study_level(self, clips, study_id)` | Combines the clips of each study into one study vector | No. Without it, the probes use the mean of each study's clip features |

To add your model, follow these steps:

1. Create `eval/embed/mymodel.py` with your class:

   ```python
   import numpy as np
   import torch
   from echoalign.loader import read_frames


   class MyModelEmbedder:
       name = "mymodel"

       def __init__(self, device=None):
           self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
           self.model = ...  # load your model and weights here

       def preprocess(self, path):
           """Read one echo file and return one input tensor, or None to skip the clip."""
           frames = read_frames(path)  # 16 consecutive RGB PIL images
           return ...  # for example, a tensor [16, 3, 224, 224]

       @torch.inference_mode()
       def encode(self, batch):
           """Return a dict of arrays with one row per clip in the batch."""
           return dict(features=self.model(batch.to(self.device)).float().cpu().numpy())

       def study_level(self, clips, study_id):
           """Optional: one feature vector per study. This example takes the mean of the clips."""
           uniq, group = np.unique(study_id, return_inverse=True)
           sums = np.zeros((len(uniq), clips["features"].shape[1]), np.float64)
           np.add.at(sums, group, clips["features"])
           return dict(study_id=uniq, features=(sums / np.bincount(group)[:, None]).astype(np.float32))
   ```

2. Register the model in `eval/embed/__init__.py`: add `"mymodel"` to `METHODS`, and add a
   branch to `build` that returns `MyModelEmbedder(**kwargs)`.

## Datasets

To download the public datasets and write their manifests, see [DATASETS.md](DATASETS.md).
