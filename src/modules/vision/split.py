import json
import os
import shutil
import random

dataset_path = "dataset_images/"
json_file = os.path.join(dataset_path, "annotations.json")
output_base = "train_data"

for p in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(output_base, p), exist_ok=True)

with open(json_file) as f:
    data = json.load(f)

images = {img['id']: img for img in data['images']}

img_annots = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    img_name = images[img_id].get('file_name', f"ID_{img_id}")

    if img_id not in img_annots:
        img_annots[img_id] = []

    seg_data = ann.get('segmentation')

    if not seg_data:
        print(f"Erro: '{img_name}' (ID {ann['id']}) - Sem dados de segmentação.")
        continue

    try:
        if isinstance(seg_data, list) and len(seg_data) > 0:
            points = seg_data[0]

            img_w, img_h = images[img_id]['width'], images[img_id]['height']
            seg_coords = []

            for i in range(0, len(points), 2):
                x = points[i] / img_w
                y = points[i + 1] / img_h
                seg_coords.append(f"{x:.6f} {y:.6f}")

            line = f"{ann['category_id']} " + " ".join(seg_coords) + "\n"
            img_annots[img_id].append(line)
        else:
            print(f"Aviso: '{img_name}' usa formato RLE. A saltar (use polígonos no Roboflow).")

    except Exception as e:
        print(f"Erro ao processar '{img_name}': {e}")
        continue

img_ids = list(images.keys())
random.shuffle(img_ids)
split_idx = int(len(img_ids) * 0.8)

train_ids = img_ids[:split_idx]
val_ids = img_ids[split_idx:]


def move_files(ids, split):
    for id in ids:
        img_filename = images[id]['file_name']

        src_img = os.path.join(dataset_path, img_filename)
        dst_img = os.path.join(output_base, f"images/{split}", img_filename)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

        label_filename = img_filename.rsplit('.', 1)[0] + ".txt"
        with open(os.path.join(output_base, f"labels/{split}", label_filename), "w") as f:
            if id in img_annots:
                f.writelines(img_annots[id])


move_files(train_ids, 'train')
move_files(val_ids, 'val')
print("Dataset preparado com sucesso!")