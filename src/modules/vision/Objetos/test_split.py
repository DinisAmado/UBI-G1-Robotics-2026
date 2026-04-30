import cv2
import numpy as np
import os
import glob


image_files = glob.glob("train_data/images/train/*.jpg")
if not image_files:
    print("Nenhuma imagem encontrada em train_data/images/train/")
    exit()

img_path = image_files[100]
label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

print(f"A testar Imagem: {img_path}")
print(f"A testar Label: {label_path}")

img = cv2.imread(img_path)
if img is None:
    print("Erro ao ler a imagem. Verifica o caminho!")
    exit()

h, w, _ = img.shape

if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = list(map(float, line.split()))
            class_id = int(data[0])

            points = np.array(data[1:]).reshape(-1, 2)

            points[:, 0] *= w
            points[:, 1] *= h
            points = points.astype(np.int32)

            color = (0, 255, 0)
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
            cv2.putText(img, f"ID: {class_id}", (points[0][0], points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Verificacao Roboflow -> YOLO", img)
    print("Sucesso! Prime uma tecla na imagem para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Ficheiro de label não encontrado: {label_path}")