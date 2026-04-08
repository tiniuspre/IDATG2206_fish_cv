from pathlib import Path

import cv2
from detect_img import model


def get_whole_fish_boxes(
    image: cv2.typing.MatLike,
    edge_margin: int = 100,
) -> list[tuple[int, int, int, int]]:
    """Return boxes for fish fully inside the image."""
    results = model(image)
    h, w = image.shape[:2]

    boxes = []
    for *box, _conf, _cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, box)

        if (
            x1 > edge_margin
            and y1 > edge_margin
            and x2 < w - edge_margin
            and y2 < h - edge_margin
        ):
            boxes.append((x1, y1, x2, y2))

    return boxes


def expand_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int, int],
    pad: int = 40,
) -> tuple[int, int, int, int]:
    """Expand a box by pad pixels on all sides, staying inside the image."""
    x1, y1, x2, y2 = box
    h, w = image_shape[:2]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return x1, y1, x2, y2


def extract_all_frames_from_video(
    video_path: str,
    save_img_to_disk: bool = False,  # noqa: FBT001 FBT002
) -> list[cv2.typing.MatLike]:
    """Extract fish crops from video frames where the whole fish is visible."""
    output_dir = Path('data/result')
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    fish_crops = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 5 == 0:
            fish_boxes = get_whole_fish_boxes(frame, edge_margin=130)

            for fish_idx, box in enumerate(fish_boxes):
                x1, y1, x2, y2 = expand_box(box, frame.shape, pad=40)
                fish_crop = frame[y1:y2, x1:x2]

                fish_crops.append(fish_crop)
                if save_img_to_disk:
                    out_path = output_dir / f'fish_{frame_idx}_{fish_idx}.jpg'
                    cv2.imwrite(str(out_path), fish_crop)

        frame_idx += 1

    cap.release()
    return fish_crops
