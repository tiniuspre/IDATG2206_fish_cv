from pathlib import Path

import cv2

from fish_cv.constants import D_A_VIDEO, D_B_VIDEO, H_B_VIDEO, L_B_VIDEO
from fish_cv.geometry import find_left_right_points
from fish_cv.measurement import calculate_real_length, pixel_distance
from fish_cv.segmentation import apply_mask, create_fish_mask, keep_largest_contour

from .detect_img import model

file_path = Path(__file__).parent.parent


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


def extract_all_frames_from_video(
    video_path: str,
    save_img_to_disk: bool = False,  # noqa: FBT001 FBT002
) -> list[cv2.typing.MatLike]:
    """Extract whole video frames where the whole fish is visible."""
    output_dir = str(file_path / 'data' / 'result')


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f'Cannot open video: {video_path}') # noqa: TRY003 EM102

    frame_idx = 0
    frames_with_fish = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 5 == 0:
            fish_boxes = get_whole_fish_boxes(frame, edge_margin=60)

            if fish_boxes:
                frames_with_fish.append(frame.copy())

                if save_img_to_disk:
                    out_path = output_dir + f'fish_frame_{frame_idx}.jpg'
                    cv2.imwrite(str(out_path), frame)

        frame_idx += 1

    cap.release()
    return frames_with_fish


def find_average_length_from_video(video_path: str) -> (float, int):
    """Process video frames to find average fish length and count."""
    all_frames = extract_all_frames_from_video(video_path, save_img_to_disk=True)
    count = 0
    img_measures = []

    for f in all_frames:
        fish_mask = create_fish_mask(f)
        clean_mask = keep_largest_contour(fish_mask)

        left, right = find_left_right_points(clean_mask)
        result = apply_mask(f, clean_mask)

        h_a = pixel_distance(left, right)
        l_a = calculate_real_length(h_a, H_B_VIDEO, L_B_VIDEO, D_A_VIDEO, D_B_VIDEO)

        img_measures.append(l_a)

        cv2.circle(result, left, 10, (0, 0, 255), -1)
        cv2.circle(result, right, 10, (255, 0, 0), -1)

        cv2.imwrite(str(file_path / 'data' / 'result' / f'{count}.jpg'), result)

        count += 1

    return sum(img_measures) / len(img_measures), count
