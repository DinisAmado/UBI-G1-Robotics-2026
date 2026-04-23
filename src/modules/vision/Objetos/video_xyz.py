import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Configuração da RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Iniciar Pipeline
profile = pipeline.start(config)
# Obter a escala do sensor de profundidade (ex: 0.001 para milímetros)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Obter parâmetros intrínsecos da lente (crucial para X e Y reais)
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Alinhamento: Força o mapa de profundidade a "encaixar" na imagem de cor
align = rs.align(rs.stream.color)

model = YOLO('runs/train/weights/best.pt')

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame: continue

        # Converter para numpy com .copy() para evitar o erro 139
        color_image = np.asanyarray(color_frame.get_data()).copy()

        results = model.predict(color_image, conf=0.5, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Centro do objeto detetado pelo YOLO
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # 1. Obter a profundidade (Z) nativa do sensor laser
                dist = depth_frame.get_distance(u, v)  # Distância em metros

                if dist > 0:
                    # 2. Calcular Coordenadas 3D Reais (X, Y, Z) usando a SDK
                    # Esta função usa a distorção da lente para dar o valor real
                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], dist)
                    x, y, z = point_3d  # Coordenadas em metros

                    # Desenhar no ecrã
                    cv2.putText(color_image, f"XYZ: {x:.2f}, {y:.2f}, {z:.2f}m", (u, v - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)

        cv2.imshow("Nativa RealSense 3D", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    pipeline.stop()