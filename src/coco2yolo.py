import json
import os

# Папки
json_path = 'InsPLAD-det/annotations/instances_val.json'
img_dir = 'InsPLAD-det/val/images'
label_dir = 'InsPLAD-det/val/labels'

os.makedirs(label_dir, exist_ok=True)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Сопоставить image_id -> file_name
id2name = {img['id']: img['file_name'] for img in data['images']}

# COCO bbox: [x_min, y_min, width, height]
def coco2yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h

# Собираем аннотации для каждого изображения
img_wh = {img['id']: (img['width'], img['height']) for img in data['images']}
anns_per_img = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    anns_per_img.setdefault(img_id, []).append(ann)

for img_id, anns in anns_per_img.items():
    file_name = id2name[img_id]
    img_w, img_h = img_wh[img_id]
    label_lines = []
    for ann in anns:
        cat_id = ann['category_id'] - 1  # YOLO: class_id с 0, COCO: с 1
        bbox = ann['bbox']
        x_center, y_center, w, h = coco2yolo(bbox, img_w, img_h)
        label_lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    label_file = os.path.splitext(file_name)[0] + '.txt'
    with open(os.path.join(label_dir, label_file), 'w') as f:
        f.write('\n'.join(label_lines))