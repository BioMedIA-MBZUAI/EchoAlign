# Public dataset access and preparation

This guide explains how to obtain and format each public echocardiography dataset evaluated in the EchoAlign paper.

---

## 1. MIMIC-IV-Echo (DICOM)

MIMIC-IV-Echo is a large-scale public dataset of transthoracic echocardiogram (TTE) studies in DICOM format.

- **Dataset URL:** [PhysioNet MIMIC-IV-Echo v1.0](https://physionet.org/content/mimic-iv-echo/1.0/). The paper's labels are built from v1.0. A later version (v1.0.1) exists; we did not test it.
- **Access requirements:** Credentialed PhysioNet account, completion of human research participant training (for example, CITI "Data or Specimens Only Research"), and a signed data use agreement.
- **Download:** follow the download instructions in the "Files" section of the dataset page.
- **Structure:** Files are stored as standard DICOM files (`.dcm`), organized by patient and study directories.

---

## 2. CardioBench

[CardioBench](https://arxiv.org/abs/2510.00520) groups 11 tasks from eight public datasets. Its code is included as the submodule `benchmarks/CardioBench`. Download each dataset from its provider:

| Dataset | Tasks (task ID) | Format | Download |
|---|---|---|---|
| EchoNet-Dynamic | Ejection fraction (`ejection_fraction_public`) | Video (`.avi`) | [echonet.github.io/dynamic](https://echonet.github.io/dynamic/) |
| EchoNet-Pediatric | Ejection fraction (`ejection_fraction_pediatric_public`) | Video (`.avi`) | [echonet.github.io/pediatric](https://echonet.github.io/pediatric/) |
| EchoNet-LVH | Interventricular septum, LV internal diameter, and LV posterior wall thickness at end-diastole (`ivsd_lvh_public`, `lvidd_lvh_public`, `lvpwd_lvh_public`) | Video (`.avi`) | [echonet.github.io/lvh](https://echonet.github.io/lvh/) |
| CAMUS | Ejection fraction (`ejection_fraction_camus_public`) | NIfTI (`.nii.gz`) | [CREATIS CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) |
| TMED-2 | Aortic stenosis (`aortic_stenosis_tmed2_public`) | Images (`.png`) | [tmed.cs.tufts.edu](https://tmed.cs.tufts.edu/tmed_v2.html) |
| CardiacNet | Atrial septal defect (`asd_cardiacnet_public`), pulmonary arterial hypertension (`pah_cardiacnet_public`) | NIfTI (`.nii`, `.nii.gz`) | [Kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/abnormcardiacechovideos) |
| SegRWMA | Regional wall motion abnormality (`rwma_public`) | NIfTI (`.nii.gz`) | [Kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/regional-wall-motion-abnormality-echo) |
| HMC-QU | ST-elevation myocardial infarction (`stemi_hmc_qu_public`) | Video (`.avi`, `.mp4`) | [Kaggle](https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset) |

CAMUS, EchoNet-Dynamic, and EchoNet-Pediatric are also used for [segmentation](README.md#evaluate-the-models).

The repository includes the CardioBench split files. Build the two label files from your downloads with `scripts/make_cardiobench_labels.py`:

- `data/splits/<dataset>.csv`: CardioBench's split of each dataset (its official split, or the split in `benchmarks/CardioBench/data/splits/`).
- `data/labels/cardiobench_probe.csv`: the label of each study for the 11 tasks, for linear probing.
- `data/labels/cardiobench.csv`: the TEST studies of the 5 classification tasks, for zero-shot.

`scripts/make_cardiobench_splits.py` rebuilds the nine split files from each dataset's official split files and the CardioBench submodule. EchoNet-Pediatric ships 10 folds and no split, so fold 0 is TEST, fold 1 is VAL, and folds 2-9 are TRAIN. `scripts/make_cardiobench_labels.py` writes the two label files from the raw datasets and these split files. Both read the dataset folders from `configs/datasets.yaml`.

You don't need all nine datasets. By default, `scripts/make_cardiobench_labels.py` and `scripts/make_manifests.py` use every dataset whose folder is set, and `--datasets` picks a subset (for example `--datasets cardiacnet_asd cardiacnet_pah`). The `cardiobench` cohort then holds only those datasets and their tasks. Bootstrap draws resample every study in the label file, so the confidence intervals of a subset can differ slightly from those on all nine datasets.

To get CardioBench's own split files and evaluation code, initialize the submodule:

```bash
git submodule update --init benchmarks/CardioBench
```

---

## 3. Prepare the datasets

To prepare the datasets, follow these steps:

1. Download each dataset from the links in [MIMIC-IV-Echo](#1-mimic-iv-echo-dicom) and [CardioBench](#2-cardiobench).
2. Set the folder of each downloaded dataset in `configs/datasets.yaml`. To keep your paths out of git,
   write them in `configs/datasets.local.yaml` instead (same keys); the scripts read it when it exists.
3. Build the MIMIC-IV-Echo split and label files and the CardioBench label files from your
   downloads. The repository does not include them; the MIMIC-IV-Echo files are derived from
   credentialed PhysioNet data:

   ```bash
   python scripts/make_mimic_labels.py --mimic-echo <MIMIC-IV-Echo folder>
   python scripts/make_cardiobench_labels.py
   ```

   The script defines every rule: the patient split, the value sets of each finding,
   the regression ranges and the severity grades. A grade with no study is not a level, a graded
   finding needs at least three levels, and every level must occur in TRAIN, VAL and TEST. When a
   grade is missing from a split, the top two grades merge once; a finding that still fails, or whose
   merged grade has fewer than five TEST studies, is left out.

4. Write the manifests:

   ```bash
   python scripts/make_manifests.py
   ```

   The script writes one manifest per dataset, `data/manifests/<dataset>.csv`, with the paper's
   clips, studies, and splits:

   | Dataset | Manifest | Clips | Studies |
   |---|---|---:|---:|
   | MIMIC-IV-Echo | `mimic_iv_echo.csv` | 525,319 | 7,243 |
   | EchoNet-Dynamic | `echonet_dynamic.csv` | 10,030 | 10,030 |
   | EchoNet-Pediatric | `echonet_pediatric.csv` | 7,810 | 4,467 |
   | EchoNet-LVH | `echonet_lvh.csv` | 11,766 | 11,766 |
   | CAMUS | `camus.csv` | 1,000 | 500 |
   | TMED-2 | `tmed2.csv` | 17,270 | 598 |
   | CardiacNet ASD | `cardiacnet_asd.csv` | 228 | 228 |
   | CardiacNet PAH | `cardiacnet_pah.csv` | 476 | 476 |
   | SegRWMA | `segrwma.csv` | 515 | 190 |
   | HMC-QU | `hmc_qu.csv` | 322 | 322 |

   When every dataset except MIMIC-IV-Echo is set, the script also writes `cardiobench.csv`, which
   joins the nine CardioBench manifests (CardiacNet has one per task; IDs prefixed with the dataset name).
5. Check that each dataset is read in the right orientation and colors: write a manifest with a few clips and run
   `python scripts/check_inputs.py --manifest <that CSV> --out-dir results/input_check`. For each clip, it saves the first and middle frame
   each model receives, next to the same frames from the model's own code where available.

## 4. File formats

EchoAlign reads DICOM, video, NIfTI, single-image, and frame-folder inputs. For the formats, the default
preprocessing, and every loader option, see [Loading echo clips](echoalign/README.md#loading-echo-clips).

### Manifest CSV (`data/manifests/<dataset>.csv`)

Each manifest has these columns:

```csv
clip_id,study_id,cluster_id,split,path
ES0001_A2C,ES0001_A2C,ES0001,TRAIN,/data/HMC_QU/videos/A2C/ES0001_CH2_1.avi
```

- `cluster_id`: the unit the bootstrap resamples. It is the patient for MIMIC-IV-Echo and
  HMC-QU, and the study for the other datasets. EchoNet-Pediatric has no patient field, so its study is
  the two-part file-name group.
- `split`: `TRAIN`, `VAL`, or `TEST`.

For your own data, write a CSV with the same columns and add a cohort with its manifest, label files, and seed key to `configs/cohorts.yaml`.

### Label CSV (`data/labels/<dataset>.csv`)

A zero-shot label file has one row per study and task, with a label of 0 or 1:
```csv
study_id,task,label,cluster_id
study_01,aortic_stenosis,1,patient_101
study_01,mitral_regurgitation,0,patient_101
study_02,aortic_stenosis,0,patient_102
```
`cluster_id` is optional. The bootstrap resamples whole clusters, so all studies of one patient
stay together. Copy it from the manifest; without it, each study is its own cluster.

For linear probing, `data/labels/<dataset>_probe.csv` adds a `split` column (`TRAIN`, `VAL`, `TEST`) and, optionally, `mode` (`classification`, `regression`, or `ordinal`, where `label` is a grade from 0 for the lowest).

## 5. Private cohort

The private cohort is not distributed, and the scripts that build its files from the iCardio data are not in the repository. The figures that describe the private cohort read aggregate tables that the authors keep.
