import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Tenta usar a versão macosx se instalaste
# pip install pyrealsense2-macosx

pipeline = rs.pipeline()
config = rs.config()

# Pedir o mínimo possível para evitar o Signal 11
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

model = YOLO('runs/train/weights/best.pt')

try:
    pipeline.start(config)
    # Alinhamento é pesado, mas necessário para Pose
    align = rs.align(rs.stream.color)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame: continue

        # CONVERSÃO SEGURA PARA NUMPY
        img = np.asanyarray(color_frame.get_data()).copy()

        # YOLO - fazemos o predict apenas a cada 2 frames se necessário para poupar CPU
        results = model.predict(img, conf=0.6, verbose=False)

        for r in results:
            for box in r.boxes:
                # Coordenadas do Bounding Box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Obter Distância (Z)
                z = depth_frame.get_distance(u, v)

                if z > 0:
                    # Aqui calcularias o X, Y, Z e R, P, Y
                    cv2.putText(img, f"Z: {z:.2f}m", (u, v), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("6DoF Test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()