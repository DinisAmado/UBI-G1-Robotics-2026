import cv2
from ultralytics import YOLO
import sys

# 1. Carregar o modelo YOLO
try:
    # Se o ficheiro 'best.pt' não estiver na pasta, podes usar 'yolov8n.pt' para testar
    model = YOLO('runs/train/weights/best.pt')
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    sys.exit()

# 2. Iniciar captura da câmara do Mac
# O índice 0 é geralmente a câmara integrada (FaceTime HD Camera)
cap = cv2.VideoCapture(0)

# Verificar se a câmara abriu corretamente
if not cap.isOpened():
    print("Erro: Não foi possível aceder à câmara do Mac.")
    sys.exit()

print("Câmara do Mac iniciada. Prime 'q' para sair.")

try:
    while True:
        # Ler o frame da câmara
        ret, frame = cap.read()

        if not ret:
            print("Erro ao receber frame.")
            break

        # 3. Predição com YOLO
        # stream=True é mais eficiente para vídeo em direto
        results = model.predict(source=frame, conf=0.8, verbose=False)

        # 4. Obter o frame anotado
        # results[0].plot() desenha as caixas e nomes das classes
        annotated_frame = results[0].plot()

        # Mostrar o resultado numa janela
        cv2.imshow("YOLOv8 - Webcam Mac", annotated_frame)

        # Sair com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Libertar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Programa encerrado.")