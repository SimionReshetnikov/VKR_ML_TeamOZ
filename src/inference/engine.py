import numpy as np
import cv2
from ultralytics import YOLO

class InferenceEngine:
    def __init__(self, config=None):
        if config is None:
            from utils.config import Config
            config = Config()
        self.config = config if hasattr(config, "get") else None
        self.update_params()

    def update_params(self):
        cfg = self.config.config if self.config else {}
        self.detector_ckpt = cfg.get("detector_ckpt", "checkpoints/detector/weights/best.pt")
        self.classifier_ckpt = cfg.get("classifier_ckpt", "checkpoints/classifier/weights/best.pt")
        self.device = cfg.get("device", "cuda")
        self.defect_threshold = float(cfg.get("defect_threshold", 0.7))
        self.good_threshold = float(cfg.get("good_threshold", 0.5))

        print(f"Загрузка детектора: {self.detector_ckpt}")
        self.detector = YOLO(self.detector_ckpt)
        print(f"Загрузка классификатора: {self.classifier_ckpt}")
        self.classifier = YOLO(self.classifier_ckpt)

        self.detector_classes = [
            "yoke", "yoke suspension", "spacer", "stockbridge damper", "lightning rod shackle",
            "lightning rod suspension", "polymer insulator", "glass insulator", "tower id plate",
            "vari-grip", "polymer insulator lower shackle", "polymer insulator upper shackle",
            "polymer insulator tower shackle", "glass insulator big shackle", "glass insulator small shackle",
            "glass insulator tower shackle", "spiral damper", "sphere"
        ]
        self.classifier_classes= [
            "yoke-suspension_good", "yoke-suspension_rust", "vari-grip_good", "vari-grip_rust",
            "polymer-insulator-upper-shackle_rust", "vari-grip_bird-nest", "polymer-insulator-upper-shackle_good",
            "glass-insulator_missing-cap", "lightning-rod-suspension_good", "lightning-rod-suspension_rust",
            "glass-insulator_good"
        ]

    def reload_config(self, config):
        self.config = config
        self.update_params()

    def infer(self, image):
        results = []
        det_out = self.detector.predict(image, device=self.device, verbose=False)[0]
        boxes = det_out.boxes.xyxy.cpu().numpy()
        scores = det_out.boxes.conf.cpu().numpy()
        classes = det_out.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            obj_class_id = classes[i]
            obj_conf = float(scores[i])
            obj_class_name = self.detector_classes[obj_class_id]

            roi = image[y1:y2, x1:x2]
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                continue

            cls_out = self.classifier.predict(roi, device=self.device, verbose=False)[0]
            cls_probs = cls_out.probs.data.cpu().numpy()
            cls_id = int(np.argmax(cls_probs))
            defect_class = self.classifier.model.names[cls_id] if hasattr(self.classifier.model, 'names') else self.classifier_classes[cls_id]
            defect_conf = float(cls_probs[cls_id])
            print
            # --- Фильтрация по good_threshold ---
            if "good" in defect_class:
                if defect_conf < self.good_threshold or obj_conf < self.good_threshold:
                    continue  # Фильтруем, если хотя бы одна из вероятностей ниже good_threshold
            else:
                if defect_conf < self.defect_threshold or obj_conf < self.defect_threshold:
                    continue  # Фильтруем, если хотя бы одна из вероятностей ниже defect_threshold

            results.append({
                'bbox': [x1, y1, x2, y2],
                'object_class': obj_class_name,
                'object_conf': obj_conf,
                'defect_class': defect_class,
                'defect_conf': defect_conf
            })

        return results

    def draw_results(self, image, results, draw_heatmap=False):
        img = image.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            label = f"{res['object_class']} ({res['object_conf']:.2f})"
            label2 = f"{res['defect_class']} ({res['defect_conf']:.2f})"
            is_defect = "good" not in res['defect_class']
            color = (0, 0, 255) if is_defect else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(img, label2, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if draw_heatmap:
            overlay = img.copy()
            for res in results:
                x1, y1, x2, y2 = res['bbox']
                alpha = min(0.5, max(0.1, res['object_conf']))
                is_defect = "good" not in res['defect_class']
                color = (0, 0, 255) if is_defect else (0, 255, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
    
if __name__ == '__main__':
    import argparse
    from utils.config import Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='Путь к изображению для теста')
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()

    config = Config(args.config)
    engine = InferenceEngine(config)
    img = cv2.imread(args.img)
    results = engine.infer(img)
    print("Результаты:")
    for r in results:
        print(r)
    img_vis = engine.draw_results(img, results, draw_heatmap=True)
    cv2.imwrite('result_vis.jpg', img_vis)
    print("Сохранено в result_vis.jpg")