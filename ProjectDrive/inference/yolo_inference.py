import cv2
from ultralytics import YOLO

class YOLOInference:
    def __init__(self, model_path, conf_threshold=0.4):
        """
        Initialize YOLOv5
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = {0: "car", 1: "truck"}

    def predict_and_visualize(self, image, alpha=0.4):
        """
        Run object detection and return image.
        """
        results = self.model.predict(image, verbose=False)[0]
        overlay = image.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.conf_threshold or cls_id not in self.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{self.target_classes[cls_id]} {conf:.2f}"
            color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


if __name__ == "__main__":
    img_path = "../../BDD100K\images\\test\cf6079b6-e6f1f4fc.jpg"
    image = cv2.imread(img_path)

    yolo_detector = YOLOInference(model_path="../models/best_YOLO.pt")
    result_img = yolo_detector.predict_and_visualize(image)

    cv2.imshow("YOLOv5 Detection", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
