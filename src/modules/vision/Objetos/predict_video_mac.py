import cv2
import numpy as np
from ultralytics import YOLO
import sys

# 1. Carregar o modelo
try:
    model = YOLO('runs/train/weights/best.pt')
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    sys.exit()

# 2. Cores Personalizadas (BGR) e Definição de Opacidade
custom_colors = {
    'bola': (0, 255, 0),  # Verde
    'pasta': (128, 0, 128),  # Roxo
    'cubo': (0, 0, 255)  # Vermelho
}
ALPHA = 0.4  # Opacidade: 0.0 (invisível) a 1.0 (totalmente sólido)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Criar um overlay (camada extra) para as máscaras
    overlay = frame.copy()

    # 3. Predição
    results = model.predict(source=frame, conf=0.9, verbose=False)

    for r in results:
        # --- DESENHAR SEGMENTAÇÃO COM OPACIDADE ---
        if r.masks is not None:
            for mask, box in zip(r.masks.xy, r.boxes):
                label = model.names[int(box.cls[0])]
                color = custom_colors.get(label, (255, 255, 255))

                # Desenhar a máscara no overlay
                points = np.int32([mask])
                cv2.fillPoly(overlay, points, color)

        # Aplicar o overlay ao frame original com transparência
        frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

        # --- DESENHAR BOXES E CONFIANÇA (Sem transparência para ler melhor) ---
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            color = custom_colors.get(label, (255, 255, 255))

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Caixa (Bounding Box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Texto: Classe + Confiança (ex: pasta 0.97)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv8 - Opacidade e Confianca", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()