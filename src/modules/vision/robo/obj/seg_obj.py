import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
import sys


# Carregar o modelo
model_path = 'best.pt'
print(f"Carregar modelo | Caminho: {model_path}")


try:
    model = YOLO(model_path)
    print("[OK] Modelo carregado com sucesso.")
except Exception as e:
    print(f"[ERRO] Falha ao carregar o modelo: {e}")
    sys.exit()

# Definições de visualização
custom_colors = {
    'bola': (0, 255, 0),
    'pasta': (128, 0, 128),
    'cubo': (0, 0, 255)
}
ALPHA = 0.4

# Configuração da RealSense
print("[DEBUG] Inicializando o processo da RealSense.")
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    pipeline.start(config)
    print("[OK] Câmera RealSense iniciada.")
except Exception as e:
    print(f"[ERRO] Falha ao conectar com a câmera: {e}")
    sys.exit()

print("\n[INFO] Sistema Processado (Pressione 'q' para sair)")

try:
    while True:
        # Esperar por frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Converter para numpy array
        frame = np.asanyarray(color_frame.get_data())
        overlay = frame.copy()

        # Predição
        results = model.predict(source=frame, conf=0.9, verbose=False, device='cpu')

        for r in results:
            if len(r.boxes) > 0:
                detetados = [model.names[int(c)] for c in r.boxes.cls]
                print(f"[DETECTADO] {detetados}", end='\r')

            if r.masks is not None:
                for mask, box in zip(r.masks.xy, r.boxes):
                    label = model.names[int(box.cls[0])]
                    color = custom_colors.get(label, (255, 255, 255))
                    points = np.int32([mask])
                    cv2.fillPoly(overlay, points, color)

            frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                color = custom_colors.get(label, (255, 255, 255))
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("G1 RealSense", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Sair.")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[FINALIZADO] Processo Finalizado.")