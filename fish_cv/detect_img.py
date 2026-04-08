import torch
import cv2

model = torch.hub.load('../yolo_minimal', 'custom', 'best_mult.pt', source='local', trust_repo=True)
model.conf = 0.5


def has_fish_in_image(image) -> bool:
    results = model(image)
    return len(results.xyxy[0]) > 0


if __name__ == '__main__':
    img = cv2.imread('../data/fish/fishtestimg.jpg')
    results = model(img)

    for *box, conf, cls in results.xyxy[0].cpu().numpy():
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
    print(has_fish_in_image(img))
