# IDATG2206 Fish CV — Fish length estimation (image + video)

A small computer-vision project that estimates fish length from either a **single image** or a **video**.

It uses:
- **OpenCV** for image processing/segmentation
- **NumPy** for geometry/math
- A bundled **YOLO (local Torch Hub)** model to detect frames where the fish is fully visible (used in video processing)

---

## Project structure

- `main.py` — entrypoint; choose between image (`i`) and video (`v`) processing.
- `fish_cv/` — main package:
  - `segmentation.py` — load image, create binary fish mask, keep largest contour, apply mask
  - `geometry.py` — find leftmost/rightmost mask pixels (fish endpoints)
  - `measurement.py` — pixel distance + real-world length conversion
  - `video_tool.py` — extracts frames from video and computes average fish length
  - `detect_img.py` — loads a local YOLO model via `torch.hub.load(...)`
  - `constants.py` — paths + calibration constants (distances, reference sizes)
  - `avg_measurements.py` — utilities for averaging/robust statistics across multiple measurements
  - `best_mult.pt` — trained weights for fish detection (large file)
- `data/` — input images/videos and output results (paths configured in constants)
- `yolo_minimal/` — minimal YOLO implementation used as a local Torch Hub repo

---

## Requirements

From `requirements.txt`:
- `numpy`
- `opencv-python`
- `ruff`

**Note:** video processing also imports **PyTorch** (`torch`) via `fish_cv/detect_img.py`.  
If you plan to run video mode, make sure you also install a compatible PyTorch build.

---

## Setup

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

If you want to run video processing (YOLO-based frame filtering), also install PyTorch:

```bash
pip install torch
```

(Choose the correct PyTorch command for your OS/CUDA setup if needed.)

---

## Configuration (paths + calibration)

Edit `fish_cv/constants.py` to point to your own files and calibration values.

Key settings:
- `INPUT_PATH` — input image file
- `OUTPUT_PATH` — where the result image is written
- `VIDEO_PATH` — input video file
- `H_B`, `L_B`, `D_B`, `D_A` — image calibration constants
- `H_B_VIDEO`, `L_B_VIDEO`, `D_B_VIDEO`, `D_A_VIDEO` — video calibration constants

The length conversion formula is implemented in `fish_cv/measurement.py`:

- Measure fish length in pixels (`h_a`)
- Convert to real-world length:

`l_a = l_b * (h_a / h_b) * (d_a / d_b)`

Where:
- `h_b` is a reference length in pixels
- `l_b` is the real reference length (e.g., cm)
- `d_a` / `d_b` are camera-to-fish / camera-to-reference distances

---

## Usage

Run the program:

```bash
python main.py
```

You’ll be prompted:

- Enter `i` for image processing
- Enter `v` for video processing

### Image mode (`i`)
Pipeline:
1. Load image from `INPUT_PATH`
2. Create fish mask using Otsu thresholding + morphology
3. Keep largest contour
4. Find left/right endpoints in the mask
5. Compute length in pixels and convert to cm
6. Save output image to `OUTPUT_PATH` with endpoints marked

Output:
- Logs printed to console (pixel endpoints, lengths)
- Result image saved to `data/result/result.png` (by default)

### Video mode (`v`)
Pipeline:
1. Open video from `VIDEO_PATH`
2. Sample every 5th frame
3. Use YOLO (`fish_cv/detect_img.py`) to keep only frames where the fish is fully inside the frame
4. For those frames:
   - Segment fish
   - Measure length
   - Save annotated frames to `fish_cv/data/result/`
5. Compute and log the **average length** across accepted frames

Output:
- Annotated frames written to `fish_cv/data/result/`
- Average length + number of frames processed logged to console

---

## Output files

Depending on mode, outputs are written to:
- `data/result/` (image mode output path by default)
- `fish_cv/data/result/` (video tool output frames)

If you don’t see output where expected, check `fish_cv/constants.py` and `fish_cv/video_tool.py` for output paths.

---

## Development

### Lint/format (ruff)

```bash
ruff check .
ruff format .
```

Ruff config is in `pyproject.toml`.

---

## Notes / troubleshooting

- If segmentation fails with “No contours found” or “No segmented object found”, the thresholding may not work for your lighting/background. Try adjusting:
  - `KERNEL_SIZE` in `fish_cv/constants.py`
  - `kernel_size` in `create_fish_mask(...)` (see `fish_cv/segmentation.py`)
- If video mode fails at model loading, ensure:
  - `torch` is installed
  - the `yolo_minimal/` directory exists and is intact
  - weights file exists (`fish_cv/best_mult.pt`)

---

