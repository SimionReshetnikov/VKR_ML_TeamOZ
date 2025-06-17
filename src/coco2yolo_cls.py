import os
import shutil

src_dir = 'InsPLAD-fault/supervised_fault_classification/defect_supervised'
dst_dir = 'InsPLAD-fault/supervised_fault_classification/defect_supervised/all_classes'

for obj_type in os.listdir(src_dir):
    obj_path = os.path.join(src_dir, obj_type)
    if not os.path.isdir(obj_path):
        continue
    for split in ['train', 'val']:
        split_path = os.path.join(obj_path, split)
        if not os.path.exists(split_path):
            continue
        for defect_type in os.listdir(split_path):
                    defect_path = os.path.join(split_path, defect_type)
                    if not os.path.isdir(defect_path):
                        continue
                    # Новый класс: objtype_defect
                    new_class = f"{obj_type}_{defect_type}"
                    target_dir = os.path.join(dst_dir, split, new_class)
                    os.makedirs(target_dir, exist_ok=True)
                    for img in os.listdir(defect_path):
                        src_img = os.path.join(defect_path, img)
                        dst_img = os.path.join(target_dir, img)
                        shutil.copy2(src_img, dst_img)