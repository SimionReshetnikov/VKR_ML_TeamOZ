import json
import os

DEFAULT_CONFIG = {
    "detector_ckpt": "checkpoints/detector/weights/best.pt",
    "classifier_ckpt": "checkpoints/classifier/weights/best.pt",
    "device": "cuda",
    "defect_threshold": 0.7,
    "good_threshold": 0.5
}

class Config:
    def __init__(self, path="config.json"):
        self.path = path
        self.config = DEFAULT_CONFIG.copy()
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.config.update(json.load(f))
            except Exception as e:
                print(f"Ошибка загрузки config.json: {e}")

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка сохранения config.json: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save()