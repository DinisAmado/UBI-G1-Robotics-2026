from ultralytics import YOLO
import os
import cv2

modelo_path = 'runs/train/weights/best.pt'
imagem_teste = 'train_data/images/val/IMG_1439_MOV-0001_jpg.rf.VVdvgXTY73oJNJ80MtHt.jpg'
pasta_saida = 'Testes'

os.makedirs(pasta_saida, exist_ok=True)

model = YOLO(modelo_path)

results = model.predict(source=imagem_teste, conf=0.5)

for r in results:
    im_array = r.plot()

    nome_ficheiro = os.path.basename(imagem_teste)
    save_path = os.path.join(pasta_saida, f"segmentada_{nome_ficheiro}")

    # Guardar usando OpenCV
    cv2.imwrite(save_path, im_array)

    print(f"✅ Sucesso! Imagem guardada em: {save_path}")

cv2.imshow("Resultado", im_array)
cv2.waitKey(0)
cv2.destroyAllWindows()