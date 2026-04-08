import cv2
import torch

model = torch.hub.load(
    '../yolo_minimal', 'custom', 'best_mult.pt', source='local', trust_repo=True
)
model.conf = 0.5


def has_fish_in_image(image: cv2.typing.MatLike) -> bool:
    """Checks if there is a fish in the image."""
    results = model(image)
    return len(results.xyxy[0]) > 0


def is_whole_fish_in_image(image: cv2.typing.MatLike, margin: int = 10) -> bool:
    """Checks if the whole fish is in the image, allowing for a margin."""
    results = model(image)
    h, w = image.shape[:2]
    for *box, _conf, _cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        if x1 > margin and y1 > margin and x2 < w - margin and y2 < h - margin:
            return True
    return False


if __name__ == '__main__':
    img = cv2.imread('../data/fish/fishtestimg.jpg')
    results = model(img)

    for *box, conf, _cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f'fish {conf:.2f}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite('../data/result/fish_detection.jpg', img)
    print(has_fish_in_image(img))  # noqa: T201
    print(is_whole_fish_in_image(img))  # noqa: T201
