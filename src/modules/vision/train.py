from ultralytics import YOLO
import torch
import os


def train_model():

    os.makedirs('Modelos', exist_ok=True)
    os.makedirs('Resultados', exist_ok=True)

    model = YOLO('yolo11n-seg.pt')

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Treino a correr em: {device}")

    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project='Resultados',
        name='treino_g1',
        exist_ok=True
    )


    import shutil
    src = 'Resultados/treino_g1/weights/best.pt'
    dst = 'Modelos/g1_vision_v1.pt'

    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Modelo final guardado em: {dst}")


if __name__ == '__main__':
    train_model()