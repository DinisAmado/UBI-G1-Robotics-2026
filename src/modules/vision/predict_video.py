from ultralytics import YOLO
import cv2
import torch

modelo_path = 'runs/train/weights/best.pt'

model = YOLO(modelo_path)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model.to(device)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível aceder à câmara.")
    exit()

print("Vídeo iniciado. Prime 'q' para sair.")

while True:
    success, frame = cap.read()

    if not success:
        print("Falha ao ler o frame da câmara.")
        break

    results = model.predict(source=frame, conf=0.5, stream=True)

    for r in results:
        annotated_frame = r.plot()

        cv2.imshow("YOLOv11 - G1 Vision Real-Time", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()