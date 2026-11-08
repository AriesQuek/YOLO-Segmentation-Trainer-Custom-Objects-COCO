# YOLO Segmentation Trainer: Custom Objects + COCO

Two Google Colab notebooks that train a **YOLO11n-seg** instance-segmentation model on your own objects, optionally together with chosen COCO classes. Each dataset only needs to be labelled for its own object; the notebooks cross-label the other objects, and you approve them by hand.

| Notebook | What it does | Output |
|---|---|---|
| `program1_custom_objects.ipynb` | Merges your own single-object datasets, trains, cross-labels each dataset with the other objects, then retrains | `custom/vN_<K>objects_<date>/vN_<K>objects_<date>.pt` |
| `program2_custom_plus_coco.ipynb` | Takes a finished Program 1 model, adds the COCO classes you choose, cross-labels both ways, and trains from `yolo11n-seg.pt` | `full/vN_<K>objects_<date>/vN_<K>objects_<date>.pt` |

Run Program 1 whenever your own objects or datasets change. Run Program 2 after it to get the combined model.

---

## Features

- **Any number of custom objects.** Add a line to the settings and run the notebook again.
- **Two-way cross-labelling.** The model looks for missing objects in every dataset, so a staircase in a tactile-paving photo gets labelled instead of being treated as background.
- **Review by hand in Google Drive.** Delete the wrong preview images; the ones you keep are applied.
- **Recovers from Colab disconnects.** Progress is saved to Drive, so you can **Run all** again: finished steps are skipped, and training continues from the last completed epoch.
- **Versioned outputs:** `v1_…`, `v2_…`. Old versions are kept for comparison.
- **Only the COCO images you need.** It downloads images for your chosen classes instead of the whole 118k-image dataset.
- **Stable class numbers.** Custom objects come first in the order you list them, then the COCO classes.

---

## Requirements

- A Google account with Google Drive. You need at least 5 GB free; more if you use many COCO images.
- Google Colab with a **T4 GPU** runtime. The free tier works.
- Datasets in **YOLO segmentation format**. A Roboflow "YOLOv11 / YOLOv8 segmentation" export works as-is.

Colab installs Ultralytics by itself; nothing needs installing on your computer.

---

## 1. Prepare Google Drive

Create these folders yourself:

```
MyDrive/
└── yolo/
    └── datasets/
        ├── stairs.zip
        ├── tactile_paving.zip
        └── ...            (one zip per object)
```

The notebooks create everything else:

```
yolo/
├── datasets/              your zips
├── custom/                Program 1 versions
│   └── v1_2objects_2026-11-08/
│       ├── v1_2objects_2026-11-08.pt     ← custom model
│       ├── results.png, info.json, state.json
│       ├── dataset.zip, labels_reviewed.zip
│       ├── review/previews/...           ← you review here
│       └── runs/...
└── full/                  Program 2 versions
    └── v1_16objects_2026-11-08/
        └── v1_16objects_2026-11-08.pt    ← final model
```

### Dataset zip format

Each zip holds **one object**:

```
stairs.zip
└── (any folder name)/
    ├── data.yaml
    ├── train/images/  train/labels/
    ├── valid/images/  valid/labels/     (optional: 15% of train is used if missing)
    └── test/images/   test/labels/      (optional)
```

- Label lines are YOLO polygons: `class x1 y1 x2 y2 ...`, normalised 0–1. The class number inside the zip doesn't matter, because the notebook assigns its own.
- Box-only lines (`class xc yc w h`) are handled by the `BOX_LABELS` setting.
- Large photos are shrunk to `MAX_SIDE` once, which makes training faster.

---

## 2. Program 1: custom objects

1. In Colab: **File → Upload notebook** → `program1_custom_objects.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Edit the **SETTINGS** cell:
   ```python
   DATASETS = [
       ("stairs",         "stairs.zip"),
       ("tactile_paving", "tactile_paving.zip"),
       # add new objects at the END only
   ]
   ```
4. **Runtime → Run all**, and allow Google Drive access when asked.
   - It builds the merged dataset and trains **pass 1**.
   - Then it uses that model to look for the other objects in each dataset.
   - It then **stops on purpose** with `REVIEW NEEDED`.
5. **Review:** in Google Drive, open `yolo/custom/<version>/review/previews/`. Each sub-folder (`in_stairs`, `in_tactile_paving`, …) shows the newly found objects. **Delete every wrong preview.** The ones you keep count as approved.
6. In SETTINGS, set the version name that was printed:
   ```python
   REVIEWED_VERSION = "v1_2objects_2026-11-08"
   ```
7. **Run all** again. It applies the approved labels, trains **pass 2**, and saves the final custom model.

---

## 3. Program 2: custom objects + COCO

1. Upload `program2_custom_plus_coco.ipynb` to Colab and select the **T4 GPU** runtime.
2. Edit SETTINGS: `COCO_CLASSES` (default below), and `CUSTOM_VERSION` (`"latest"` or an exact Program 1 version).
   ```python
   COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
                   "traffic light", "fire hydrant", "stop sign", "parking meter",
                   "bench", "cat", "dog"]
   ```
3. **Run all.** It shows how many images each custom object has, then stops.
4. Set `PER_CLASS_TRAIN`, the number of COCO training images per class. A value close to your custom objects' image counts keeps the classes balanced.
5. **Run all.** It downloads the COCO images and cross-labels both ways:
   - The COCO model finds COCO objects in your photos.
   - Your custom model finds your objects in the COCO photos.
   
   It then stops for review.
6. Review the previews in `yolo/full/<version>/review/previews/`, set `REVIEWED_VERSION`, and **Run all** again.
   - If there are too many previews to check, set `REVIEW = "auto"` to accept everything above the confidence threshold.
7. The final model is saved as `yolo/full/<version>/<version>.pt`.

Class order in the final model: your custom objects first (0, 1, …), then the COCO classes in the order you listed them.

---

## Settings reference

| Setting | Notebook | Default | Effect |
|---|---|---|---|
| `DATASETS` | 1 | – | Object name and zip file. The list order sets the class numbers. |
| `VERSION` | both | `"auto"` | Continues the newest unfinished version, otherwise starts the next `vN`. Or give an exact folder name. |
| `REVIEW` | both | `"manual"` | `"manual"`: approve in Drive. `"auto"`: accept all proposals. |
| `REVIEWED_VERSION` | both | `""` | Set to the version name after reviewing, to continue. |
| `BOX_LABELS` | 1 | `"convert"` | Box-only labels: `"convert"` to rectangles, `"skip_image"`, or `"stop"` (lists them for redrawing). |
| `CONF` / `CONF_*` | both | `0.35` | Cross-label confidence. Lower gives more proposals to review; higher means more objects are missed. |
| `MIN_AREA` | both | `0.002` | Ignores proposed masks smaller than 0.2% of the image. |
| `PER_CLASS_TRAIN` | 2 | `None` | COCO training images per class. You choose it each run. |
| `PER_CLASS_VAL` | 2 | `150` | COCO validation images per class. |
| `PASS1_EPOCHS` / `PASS2_EPOCHS` | 1 | `60` / `40` | Training length. Stops early after `PATIENCE` epochs without improvement. |
| `PASS2_START` | 1 | `"pass1"` | Pass 2 continues from pass 1 (faster), or `"fresh"` starts from `MODEL`. |
| `EPOCHS` | 2 | `60` | Training length for the combined model. |
| `IMGSZ` | both | `640` | Training image size. Larger is more accurate on small or distant objects but slower. |
| `MAX_SIDE` | 1 | `640` | Photos are shrunk to this once. It must be ≥ `IMGSZ`. |
| `BATCH` / `WORKERS` | both | `32` / `2` | Speed and memory only. |
| `MODEL` | both | `yolo11n-seg.pt` | Starting weights. The nano size is meant for small devices. |
| `TIMEZONE` | both | `Asia/Singapore` | Used for the date in version names. |

---

## Disconnects and resuming

Free Colab sessions can end early. When that happens:

1. Reconnect, select the T4 GPU runtime, and **Run all**.
2. The notebook finds the unfinished version and does the following:
   - restores the dataset from Drive;
   - skips the finished steps;
   - resumes training from `runs/<stage>/weights/last.pt`.

If you change `DATASETS` (or the COCO classes) while a version is unfinished, the notebook stops instead of mixing data. Either restore the settings, or delete or rename the unfinished version folder.

---

## Testing a trained model

```bash
pip install ultralytics
yolo segment predict model=v1_16objects_2026-11-08.pt source=path/to/image_or_folder
yolo segment predict model=v1_16objects_2026-11-08.pt source=0 show=True   # webcam
```

---

## Licenses and credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) is licensed under **AGPL-3.0**. Check its terms before using trained models in closed-source or commercial products.
- The [COCO dataset](https://cocodataset.org/) annotations are licensed under **CC BY 4.0**. The images come from Flickr under their own terms. This repository contains no COCO data; the notebooks download it at run time.
- Your own datasets keep their original licenses. Check the license of each Roboflow dataset you use before publishing one.
